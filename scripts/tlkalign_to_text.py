import sys

def align(in_f):
    with open(in_f,"r") as fichero:
        words = []
        current_word = None
        current_pos = None
        for line in fichero:
            if line.startswith('"'):
                continue
            elif line.startswith('.'):
                continue
                
            else:
                fStartPos, fDur, fSym, fPos, FWord = line.split()
                
                if int(fPos) == 0:
                    words.append(FWord)
                                      
    print(" ".join(words))   
        
if __name__ == "__main__":
    align(sys.argv[1])

