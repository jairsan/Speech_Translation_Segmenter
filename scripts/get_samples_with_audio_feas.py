#!/usr/bin/env python3

import sys

def load_vocab(fn):
    ret= set(['</s>','<unk>'])
    for l in open(fn):
        l= l.strip()
        ret.add(l)
    return ret

def filter_vocab(words,vocab):
    return [w if w in vocab else '<unk>' for w in words]
    
#FIXME
if len(sys.argv)!=8:
    sys.exit('%s <maxlen> <ws> <vocab>'%sys.argv[0])
    
    
maxlen= int(sys.argv[1])
WS= int(sys.argv[2])
vocab= load_vocab(sys.argv[3])
assert maxlen>=(WS+1)
in_text_file=sys.argv[4]
in_audio_feas_file=sys.argv[5]
out_sample_file=sys.argv[6]
out_audio_feas_file=sys.argv[7]

labels= []
text= []
feas = []

feas_fill_value=[0]*3
# por cada x palabras que añado, llevar un contador y añadir las x feas
# que correspondan


with open(in_text_file) as in_text_f:
    for l in in_text_f:
        l=l.strip().split()
        if len(l) > 0:
            text.extend(l)
            labels.extend([0]*(len(l)-1))
            labels.append(1)

with open(in_audio_feas_file) as in_feas_f:
    for l in in_feas_f:
        l=l.strip().split()
        feas.append(l)



try:
    assert len(labels)==len(text)==len(feas)
except:
    print("Processing failed at ",in_text_file)
    print(len(labels),len(text),len(feas))
    exit

with open(out_sample_file,"a") as out_text_f, open(out_audio_feas_file,"a") as out_feas_f:
    hist= []
    hist_len= maxlen-WS-1
    histf=[]
    for i,(l,t,f) in enumerate(zip(labels,text,feas)):
        aux= hist[-hist_len:]
        #
        auxf = histf[-hist_len:]
        
        if len(aux)<hist_len:
            tmp=['</s>']*(hist_len-len(aux))
            tmp.extend(aux)
            aux=tmp
            #
            tmpf = [ feas_fill_value for i in range(hist_len-len(auxf))]
            tmpf.extend(auxf)
            auxf=tmpf
            
        aux2= text[i:i+WS+1]
        if len(aux2)<WS+1:
            aux2.extend(['</s>']*(WS+1-len(aux2)))
        aux.extend(aux2)
        aux= filter_vocab(aux,vocab)
        print(l,' '.join(aux),file=out_text_f)
        hist.append(t)
        
        #
        aux2f = feas[i:i+WS+1]
        if len(aux2f)<WS+1:
            aux2f.extend([ feas_fill_value  for i in range(WS+1-len(aux2f)) ] )
        auxf.extend(aux2f)
        histf.append(f)
        
        p = ""
        
        for feas_comp in auxf:
            p += " ".join(list(map(str,feas_comp))) + ";"
        
        print(p[:-1],file=out_feas_f)
        
        if l==1: 
            hist.append('</s>')
            #
            histf.append(feas_fill_value)
        
