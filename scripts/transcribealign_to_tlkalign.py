import sys,copy

#Features for each word -> 0:Duration of word (excl SP), 1: dur prev SP, 2: dur next SP

def get_start_end_from_name(segment_file):
 
        start_segment_file_in_milis=int(segment_file[:19].split("_")[0].replace(".", "")) / 10
        end_segment_file_in_milis=int(segment_file[:19].split("_")[1].replace(".", "")) / 10
        
        return start_segment_file_in_milis,end_segment_file_in_milis
        


def transcribe_to_align(in_f):
    with open(in_f,"r") as fichero:
        lines = fichero.readlines()
        
        
        #The clipped file which is given to the ASR system.
        segment_file = lines[0].split("/")[-1]
        

        
        _, last_segment_end = get_start_end_from_name(segment_file)
        

                
        """
        fStartPos, fDur, fSym, fPos, FWord = line.split()
        if int(fPos) -1 == current_pos:
            current_word.append((fStartPos, fDur, fSym, fPos, FWord))
            current_pos += 1
        else:
            words.append(current_word)
            current_word = [(fStartPos, fDur, fSym, fPos, FWord)]
            current_pos = 0     
        """
        

        output_lines=[]

        
        for i in range(1,len(lines)):
            line=lines[i]
            

            if line.startswith('"'):
           
                segment_file = lines[i].split("/")[-1]
                curr_segment_start, curr_segment_end = get_start_end_from_name(segment_file)
                
                #print("[II] ",curr_segment_start, curr_segment_end,last_segment_end)
                
                #Si los timestamps no coinciden, añadimos un silencio a la palabra anterior
                if curr_segment_start != last_segment_end:
                    add_to_SP = curr_segment_start - last_segment_end                    
                    pfStartPos, pfDur, pfSym, pfPos, pFWord = output_lines[-1]
                    
                    #Si no había un silencio, añado uno                    
                    if pfSym != "SP":
                        output_lines.append((str(int(pfStartPos)+int(pfDur)), add_to_SP, "SP", str(int(pfPos)+1), pFWord))
                    
                    #Si había un silencio, actualizo
                    else:
                        output_lines[-1] = (pfStartPos, str(int(pfDur)+int(add_to_SP)), pfSym, pfPos, pFWord)
                    
                    """
                    
                    if lines[i+1].split()[2] != "SP":
                        _, _, _, _, FWord_next = lines[i+1].split()
                        output_lines.append((0,add_to_SP,"SP",0,FWord_next))
                        
                        
                        currPos=-1
                        #Actualizo la fPos de la palabra a la que he añadido el silencio
                        for j in range(i+1,len(lines)):
                            nfStartPos, nfDur, nfSym, nfPos, nFWord = lines[j].split()
                            if nfPos -1 == currPos:
                                lines[i+1] = " ".join([nfStartPos, nfDur, nfSym, nfPos+1, nFWord])
                            else:
                                break
                            
                    else:
                        fStartPos, fDur, fSym, fPos, FWord = lines[i+1].split()
                        fDur += add_to_SP
                        lines[i+1] = " ".join([fStartPos, fDur, fSym, fPos, FWord])      
            """
                last_segment_end = curr_segment_end
                
            elif line.startswith('.'):
                continue
            else:   
                fStartPos, fDur, fSym, fPos, FWord = line.split()
                output_lines.append((fStartPos, fDur, fSym, fPos, FWord))
        
        
        # We reconstruct the timestamps
        count = 0
        for line in output_lines:
            fStartPos, fDur, fSym, fPos, FWord = line
            
            print(" ".join([str(count), fDur, fSym, fPos, FWord]))
            
            count+=int(fDur)
            
            
        print(".")
            
        
if __name__ == "__main__":
    transcribe_to_align(sys.argv[1])
