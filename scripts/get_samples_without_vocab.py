#!/usr/bin/env python3

import sys


if len(sys.argv)!=3:
    sys.exit('%s <maxlen> <ws> '%sys.argv[0])
maxlen= int(sys.argv[1])
WS= int(sys.argv[2])
assert maxlen>=(WS+1)

labels= []
text= []
for l in sys.stdin:
    l= l.strip().split()
    if len(l) > 0:
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
    print(l,' '.join(aux))
    hist.append(t)
    if l==1: hist.append('</s>')
