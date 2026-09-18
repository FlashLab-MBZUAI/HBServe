"""Compile calibrated kernels into the same object/KV placement path as serving."""
from hbserve.contracts import CanonicalServingBatch, SemanticOperation, HBServeError
from hbserve.gpu_profile import packed_layout


def compile_calibrated(compiler,batch):
    model,rows=compiler._validate_schedule(batch)
    provider=compiler.timing;profile=provider.profile
    key=provider.model_bindings[model.model_id]
    g=profile.document['models'][key]
    layout=packed_layout(g,profile.document['catalog'][key])
    if (model.embedding_bytes!=layout['embedding_bytes'] or model.lm_head_bytes!=layout['head_bytes'] or
        any(l.attention_weight_bytes!=layout['attention_bytes'] or l.ffn_weight_bytes!=layout['ffn_bytes'] or
            (l.is_moe and any(n!=layout['expert_bytes'] for n in l.expert_weight_bytes)) for l in model.layers)):
        raise HBServeError('bind the model to the calibrated packed weight layout before placement')
    queries=[s.token_count for s,_ in rows];contexts=[s.context_tokens_before for s,_ in rows]
    outputs=sum(s.emits_output for s,_ in rows)
    operations=[];audit={};prefix=f'serve/b{batch.batch_id}'

    def emit(role,dependencies=(),duration=0.,object_id=None,offset=0,size=0,op=None,labels=None,span_start=None,span_scale=1.):
        identifier=f'{prefix}/op{len(operations)}'
        operation=SemanticOperation(id=identifier,op=op,object_id=object_id,offset=offset,bytes=size,
            duration_ns=duration,dependencies=tuple(dict.fromkeys(dependencies)),role=role,
            span_start=span_start,span_scale=span_scale)
        operations.append(operation)
        audit[identifier]=dict(role=role,**(labels or {}))
        return identifier

    previous=emit('batch/start')
    def stage(name,c,layer=None,counts=None):
        nonlocal previous
        location=f'layer/{layer}/{name}' if layer is not None else f'tail/{name}'
        start=emit(location+'/fixed',(previous,),c['fixed_ns'],labels=dict(
            source='GPU effective operator calibration',profile_sha256=profile.digest,
            operator=name,layer=layer,flops=c['flops'],tensor_bytes=c['bytes'],
            split_identified=c['split_identified'],compute_ns_range=c['compute_ns_range']))
        compute=emit(location+'/compute',(start,),c['compute_ns'])
        memory=[];read=write=0
        def access(role,object_id,offset,size,op='R',labels=None):
            nonlocal read,write
            if not size:return
            memory.append(emit(role,(start,),object_id=object_id,offset=offset,size=size,op=op,labels=labels))
            if op=='R':read+=size
            else:write+=size
        if name=='embedding':
            for s,r in rows:
                for index in range(s.token_begin,s.token_end):
                    token,source=compiler._token_id(model,r,index)
                    access('embedding/read',model.object_id('embedding'),token*model.embedding_row_bytes,
                        model.embedding_row_bytes,labels=dict(request_id=r.request_id,token_index=index,
                            token_id=token,token_id_source=source))
        elif name in layout['attention'] and c['weight_bytes']:
            offset,size=layout['attention'][name]
            access('attention/weights',model.object_id('attention',layer),offset,size,labels=dict(layer=layer,kernel=name))
        elif name in layout['ffn']:
            offset,size=layout['ffn'][name]
            access('ffn/weights',model.object_id('ffn',layer),offset,size,labels=dict(layer=layer,kernel=name))
        elif name=='router':
            access('moe/router_weights',model.object_id('router',layer),0,layout['router_bytes'],labels=dict(layer=layer))
        elif name=='grouped_experts':
            for expert,n in enumerate(counts):
                if n:access('moe/routed_expert_weights',model.expert_object_id(layer,expert),0,
                    layout['expert_bytes'],labels=dict(layer=layer,expert=expert,routed_tokens=n))
        elif name in ('final_norm','lm_head'):
            access(name+'/read',model.object_id(name),0,c['weight_bytes'])
        if name in ('attention','kv_append'):
            for s,r in rows:
                width=model.layers[layer].kv_bytes_per_token
                if name=='attention':
                    access('attention/kv_read',model.kv_object_id(r.request_id,layer),0,
                        s.token_end*width,labels=dict(request_id=r.request_id,layer=layer,
                            context_tokens=s.context_tokens_before,query_tokens=s.token_count,
                            traffic_semantics='paged_attention_reads_history_and_new_KV'))
                else:
                    access('attention/kv_write',model.kv_object_id(r.request_id,layer),s.token_begin*width,
                        s.token_count*width,'W',dict(request_id=r.request_id,layer=layer,
                            token_begin=s.token_begin,token_count=s.token_count))
        for op,size in [('R',c['read_bytes']-read),('W',c['write_bytes']-write)]:
            if size<0:raise HBServeError(f'{name}: persistent bytes exceed the calibrated ledger')
            access('activation/'+('read' if op=='R' else 'write'),f'workspace/{model.model_id}',0,size,op,
                dict(kernel=name,layer=layer,traffic_semantics='calibrated_tensor_footprint'))
        memory_done=emit(location+'/memory',(*memory,start),span_start=start,span_scale=c['memory_service_multiplier'])
        previous=emit(location+'/complete',(compute,memory_done))

    counts=None
    for layer,l in enumerate(model.layers):
        if l.is_moe:
            counts=[0]*len(l.expert_weight_bytes)
            for s,r in rows:
                for token in range(s.token_begin,s.token_end):
                    for expert in compiler.router.experts_for(request=r,token_index=token,layer=layer,model=model):
                        counts[expert]+=1
        components=profile.components(key,queries,contexts,expert_counts=counts,output_requests=outputs)
        if layer==0:stage('embedding',components['embedding'])
        for name,c in components.items():
            if name not in ('embedding','final_norm','lm_head','sample'):stage(name,c,layer,counts)
    for name in ('final_norm','lm_head','sample'):
        if name in components:stage(name,components[name])
    emit('batch/complete',(previous,))
    return CanonicalServingBatch(schedule=batch,model_sha256=model.digest,
        request_trace_sha256=compiler.request_trace.digest,
        router_trace_sha256=compiler.router.digest if compiler.router is not None else None,
        timing_model=provider.timing_model,timing_evidence_state=provider.evidence_state,
        prefetch_depth=0,operations=tuple(operations),audit=audit)
