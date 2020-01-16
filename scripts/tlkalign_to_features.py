import sys,copy

#Features for each word -> 0:Duration of word (excl SP), 1: dur prev SP, 2: dur next SP

def align(in_f):
    with open(in_f,"r") as fichero:
        words = []
        current_word = []
        current_pos = None
        for line in fichero:
            if line.startswith('"'):
                continue
            elif line.startswith('.'):
                words.append(current_word)
                continue
                
            else:
                fStartPos, fDur, fSym, fPos, FWord = line.split()
                
                if current_pos == None:
                    current_pos = int(fPos) - 1 
                    
                if int(fPos) -1 == current_pos:
                    current_word.append((fStartPos, fDur, fSym, fPos, FWord))
                    current_pos += 1
                else:
                    words.append(current_word)
                    current_word = [(fStartPos, fDur, fSym, fPos, FWord)]
                    current_pos = int(fPos)     
        prev_SP = []
        after_SP = []
        durWord = []
             
        # We will first compute the duration of the word (excl SP) as well as the
        # duration of the SP after the word
        for word in words:
            t=0
            if word[-1][2] == "SP":
                t=int(word[-1][1])
            after_SP.append(t)
            
            
            duration=0
            for symbol in word:
                if symbol[2] != "SP":
                    duration+= int(symbol[1])
            durWord.append(duration)
            
        
        # The prev_SP can be easily computed by shifting right the after_SP list
        # We do this we eliminating the last value as well as prepending
        # the value of the SP preceding the first word
        prev_SP = copy.deepcopy(after_SP[:-1])
        
        t=0
        if word[0][2] == "SP":
            t=int(word[0][1])
        prev_SP.insert(0,t)
        
        
        assert len(prev_SP) == len(after_SP) == len(durWord)
        
        for d,p,a in zip(durWord,prev_SP,after_SP): 
            print(d,p,a,sep=" ")

        
if __name__ == "__main__":
    align(sys.argv[1])

