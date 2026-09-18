"""GPU operator calibration shared by profiling and request compilation.

The ledger counts packed tensor footprints, not DRAM counter transactions.
The profile supplies effective compute and memory service separately. Native
memory service is substituted only at execution, never added to kernel time.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import heapq
import json
import math
from pathlib import Path

from hbserve.contracts import BatchTiming, HBServeError, canonical_sha256

SCHEMA = 'hbserve.gpu_operator_profile.v3'


def attention_imbalance(queries, contexts, heads, *, sm_count=108, resident_ctas=2):
    """Critical CTA wave relative to equal-length work on the same grid.

    FA2 head-128 general paged attention uses query tiles of 64 and key
    tiles of 128. A100's 80 KiB shared memory / 128-thread CTA permits two
    resident CTAs. This affects compute service, not physical KV bytes.
    """
    if max(queries)==1: return 1.
    tasks=[math.ceil((c+min(start+64,q))/128)
           for _ in range(heads) for c,q in zip(contexts,queries) for start in range(0,q,64)]
    slots=min(sm_count*resident_ctas,len(tasks));loads=[0]*slots
    for cost in tasks:
        earliest=heapq.heappop(loads);heapq.heappush(loads,earliest+cost)
    uniform=math.ceil(len(tasks)/slots)*sum(tasks)/len(tasks)
    return max(loads)/uniform


def matrix_bytes(n, k):
    return n*k + n*(k//128)*2


def packed_layout(g, catalog):
    h,nh,kh,d,i=(g[k] for k in ('hidden','heads','kv_heads','head_dim','intermediate'))
    attention={};offset=0
    for name,size in [('qkv_projection',matrix_bytes((nh+2*kh)*d,h)),
                      ('output_projection',matrix_bytes(h,nh*d)),
                      ('input_norm',2*h),('post_norm',2*h),
                      ('rope',4*d if g['qk_norm'] else 0)]:
        attention[name]=(offset,size);offset+=size
    ffn={};foffset=0
    if not g['moe']:
        for name,size in [('gate_up_projection',matrix_bytes(2*i,h)),('down_projection',matrix_bytes(h,i))]:
            ffn[name]=(foffset,size);foffset+=size
    return dict(attention=attention,attention_bytes=offset,ffn=ffn,ffn_bytes=foffset,
        router_bytes=2*128*h if g['moe'] else 0,expert_bytes=catalog['expert_bytes'],
        embedding_bytes=2*g['vocab']*h,head_bytes=matrix_bytes(g['vocab'],h))


def operator_work(g, queries, contexts, catalog, *, expert_counts=None, output_requests=None):
    """Preserve every request's query and context, including causal prefill."""
    if len(queries)!=len(contexts) or not queries or min(queries)<1 or min(contexts)<0:
        raise HBServeError('invalid packed request shape')
    m,b=sum(queries),len(queries)
    outputs=b if output_requests is None else output_requests
    h,nh,kh,d,i=(g[k] for k in ('hidden','heads','kv_heads','head_dim','intermediate'))
    qh,kvh=nh*d,kh*d
    result={}
    def add(name,flops,reads,writes=0,weight_bytes=0,kv_read_bytes=0,kv_write_bytes=0):
        result[name]=dict(flops=int(flops),read_bytes=int(reads),write_bytes=int(writes),
            bytes=int(reads+writes),weight_bytes=int(weight_bytes),kv_read_bytes=int(kv_read_bytes),
            kv_write_bytes=int(kv_write_bytes))
    def linear(name,n,k,tokens=m):
        w=matrix_bytes(n,k)
        add(name,2*tokens*n*k,w+2*tokens*k,2*tokens*n,weight_bytes=w)
    add('embedding',0,2*m*h+8*m,2*m*h,weight_bytes=2*m*h)
    add('input_norm',0,4*m*h+2*h,2*m*h,weight_bytes=2*h)
    linear('qkv_projection',qh+2*kvh,h)
    norm=4*d if g['qk_norm'] else 0
    add('rope',0,2*m*(qh+kvh)+norm,2*m*(qh+kvh),weight_bytes=norm)
    add('kv_append',0,4*m*kvh,4*m*kvh,kv_write_bytes=4*m*kvh)
    kv=4*sum(c+q for c,q in zip(contexts,queries))*kvh
    attention_flops=4*nh*d*sum(q*c+q*(q+1)//2 for c,q in zip(contexts,queries))
    add('attention',attention_flops,kv+2*m*qh,2*m*qh,kv_read_bytes=kv)
    # FA2 paged kernels execute 64-query tiles. Pure decode uses GQA query
    # folding and split-KV; the presence of even one prefill request selects
    # the general kernel for the entire batch. Useful FLOPs alone miss this.
    decode=max(queries)==1
    heads=kh if decode else nh
    tile_context=sum(64*math.ceil((c+min(start+64,q))/128)*128
                     for c,q in zip(contexts,queries) for start in range(0,q,64))
    kernel_flops=4*heads*d*tile_context
    imbalance=attention_imbalance(queries,contexts,heads)
    result['attention'].update(kernel_flops=kernel_flops,
        timing_flops=kernel_flops*imbalance,cta_imbalance=imbalance,
        kernel_mode='decode_gqa_split' if decode else 'general_paged',query_tile=64)
    linear('output_projection',h,qh)
    add('attention_residual',0,4*m*h,2*m*h)
    add('post_norm',0,4*m*h+2*h,2*m*h,weight_bytes=2*h)
    if g['moe']:
        if expert_counts is None or len(expert_counts)!=128 or sum(expert_counts)!=8*m:
            raise HBServeError('MoE requires actual per-batch/layer expert token counts')
        if any(not isinstance(n,int) or n<0 or n>m for n in expert_counts):
            raise HBServeError('invalid expert histogram')
        bm=64
        for candidate in (8,16,32,48,64):
            bm=candidate
            if m*8/128/candidate<.9:break
        padded=sum(math.ceil(n/bm)*bm for n in expert_counts)
        router_w=2*128*h
        add('router',2*m*128*h,router_w+2*m*h+4*m*128,4*m*128+8*m*8,weight_bytes=router_w)
        w=sum(n>0 for n in expert_counts)*catalog['expert_bytes']
        scratch=2*m*8*(3*i+h)
        add('grouped_experts',6*padded*h*i,w+2*m*h+scratch,scratch+2*m*h,weight_bytes=w)
        result['grouped_experts'].update(expert_counts=list(expert_counts),padded_assignments=padded,tile_m=bm)
    else:
        linear('gate_up_projection',2*i,h)
        add('silu_multiply',0,4*m*i,2*m*i)
        linear('down_projection',h,i)
    add('ffn_residual',0,4*m*h,2*m*h)
    if outputs:
        add('final_norm',0,4*outputs*h+2*h,2*outputs*h,weight_bytes=2*h)
        linear('lm_head',g['vocab'],h,outputs)
        add('sample',0,2*outputs*g['vocab'],8*outputs)
    return result


def interpolate(axis, value):
    keys=sorted(map(int,axis))
    if value<keys[0] or value>keys[-1]:
        raise HBServeError(f'shape coordinate {value} outside measured anchors {keys}')
    if value in keys:return [(str(value),1.)]
    hi=next(k for k in keys if k>value);lo=keys[keys.index(hi)-1]
    u=(value-lo)/(hi-lo)
    return [(str(lo),1-u),(str(hi),u)]


class GPUProfile:
    def __init__(self,document):
        if document.get('schema')!=SCHEMA or document.get('pure_compute_measured') is not False:
            raise HBServeError('expected a decomposed paged GPU calibration profile')
        self.document=document
        self.digest=canonical_sha256(document)

    @classmethod
    def load(cls,path):
        return cls(json.loads(Path(path).read_text()))

    def components(self,model,queries,contexts,*,expert_counts=None,output_requests=None):
        d=self.document;g=d['models'][model];domain=d['domain']
        if (len(queries)>domain['max_batch'] or sum(queries)>domain['max_tokens'] or
            max(queries)>domain['max_query'] or max(c+q for c,q in zip(contexts,queries))>domain['max_sequence']):
            raise HBServeError('request batch outside GPU profile; collect matching measurements')
        if any(c+q>domain['short_context_max_sequence'] and q>domain['long_context_max_query']
               for c,q in zip(contexts,queries)):
            raise HBServeError('long prompt prefill is outside the measured backend domain')
        work=operator_work(g,queries,contexts,d['catalog'][model],expert_counts=expert_counts,output_requests=output_requests)
        result={}
        for op,w in work.items():
            table=d['parameters'][model][op]
            if 'all' in table:anchors=[(table['all'],1.)]
            elif 'token_regimes' in table:
                tokens=(len(queries) if op in ('final_norm','lm_head','sample') else sum(queries))
                regime='small' if tokens<=16 else 'medium' if tokens<=256 else 'large'
                anchors=[(table['token_regimes'][regime],1.)]
            else:
                anchors=[]
                for b,bw in interpolate(table,len(queries)):
                    anchors.append((table[b][w['kernel_mode']],bw))
            fixed=sum(p['fixed_ns']*v for p,v in anchors)
            compute_flops=w.get('timing_flops',w.get('kernel_flops',w['flops']))
            compute=sum(p['compute_ns_per_flop']*v for p,v in anchors)*compute_flops
            memory=sum(p['memory_ns_per_byte']*v for p,v in anchors)*w['bytes']
            multiplier=memory/w['bytes']*d['reference_hbm_bytes_per_ns']
            if multiplier<1.-1e-10:
                raise HBServeError('profile memory service is faster than its reference physical bandwidth')
            result[op]=dict(w,fixed_ns=fixed,compute_ns=compute,reference_memory_ns=memory,
                predicted_ns=fixed+max(compute,memory),
                memory_service_multiplier=max(1.,multiplier),
                split_identified=all(p['split_identified'] for p,v in anchors),
                compute_ns_range=[sum(p['compute_parameter_range'][j]*v for p,v in anchors)*compute_flops for j in (0,1)])
        return result

    def bind_model(self,model,key):
        g=self.document['models'][key];layout=packed_layout(g,self.document['catalog'][key])
        h,nh,kh,d,i=(g[k] for k in ('hidden','heads','kv_heads','head_dim','intermediate'))
        linear=2*(h*(nh+2*kh)*d+h*nh*d+(128*h+24*h*i if g['moe'] else 3*h*i))
        if model.num_layers!=g['layers'] or model.vocab_size!=g['vocab'] or any(
            l.kv_bytes_per_token!=4*kh*d or l.flops_per_token!=linear or
            l.attention_flops_per_context_token!=4*nh*d or l.is_moe!=bool(g['moe']) for l in model.layers):
            raise HBServeError('model architecture differs from the GPU calibration')
        layers=tuple(replace(l,attention_weight_bytes=layout['attention_bytes'],
            ffn_weight_bytes=layout['ffn_bytes'],router_weight_bytes=layout['router_bytes'],
            expert_weight_bytes=(layout['expert_bytes'],)*128 if g['moe'] else ()) for l in model.layers)
        provenance=dict(kind='published_descriptor',source=model.provenance['source']+'; GPU packed backend '+self.digest,
            sha256=canonical_sha256(dict(source=model.provenance,profile=self.digest,geometry=g)))
        return replace(model,layers=layers,embedding_bytes=layout['embedding_bytes'],
            lm_head_bytes=layout['head_bytes'],final_norm_bytes=2*g['hidden'],tie_word_embeddings=False,provenance=provenance)


class GPUCalibratedTimingProvider:
    timing_model='gpu_calibrated'
    evidence_state='calibrated'
    includes_compute=True

    def __init__(self,profile,model_bindings):
        self.profile=profile if isinstance(profile,GPUProfile) else GPUProfile.load(profile)
        self.model_bindings=dict(model_bindings)
        self.path=str(Path(profile).resolve()) if not isinstance(profile,GPUProfile) else None

    def canonical(self):
        return dict(type=self.timing_model,profile=self.path,profile_sha256=self.profile.digest,
            model_bindings=self.model_bindings,scope='measured operator backend; serving validation is reported separately',
            operator_validation=self.profile.document.get('operator_validation'))

    def timing_for(self,*,model,batch):
        # The compiler consumes decomposed operators. A layer-duration barrier
        # must never contain a measured kernel's memory-inclusive duration.
        return BatchTiming(layer_ns=(0.,)*model.num_layers,tail_ns=0.,
            timing_model=self.timing_model,evidence_state=self.evidence_state)
