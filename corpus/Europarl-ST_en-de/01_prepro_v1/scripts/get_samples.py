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
    
if len(sys.argv)!=4:
    sys.exit('%s <maxlen> <ws> <vocab>'%sys.argv[0])
maxlen= int(sys.argv[1])
WS= int(sys.argv[2])
vocab= load_vocab(sys.argv[3])
assert maxlen>=(WS+1)

labels= []
text= []
for l in sys.stdin:
    l= l.strip().split()
    text.extend(l)
    labels.extend([0]*(len(l)-1))
    labels.append(1)
assert len(labels)==len(text)

hist= []
hist_len= maxlen-WS-1
for i,(l,t) in enumerate(zip(labels,text)):
    aux= hist[-hist_len:]
    if len(aux)<hist_len:
        tmp= ['</s>']*(hist_len-len(aux))
        tmp.extend(aux)
        aux= tmp
    aux2= text[i:i+WS+1]
    if len(aux2)<WS+1:
        aux2.extend(['</s>']*(WS+1-len(aux2)))
    aux.extend(aux2)
    aux= filter_vocab(aux,vocab)
    print(l,' '.join(aux))
    hist.append(t)
    if l==1: hist.append('</s>')
