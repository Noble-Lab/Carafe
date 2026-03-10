import os
import re
import sys

def process_fasta(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: Input file does not exist: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
         
        missing_three_elements = 0
        missing_gn = 0
        
        for line in f_in:
            if line.startswith('>'):
                original_line = line.strip()
                
                # Split by empty with regex
                parts = re.split(r'\s+', original_line)
                protein_name = parts[0]
                
                # Split protein name by "|"
                prot_elements = protein_name.split('|')
                
                # Check for three elements
                if len(prot_elements) >= 3:
                    # Add "Cont_" to the third part
                    if not prot_elements[2].startswith("Cont_"):
                        prot_elements[2] = "Cont_" + prot_elements[2]
                    parts[0] = "|".join(prot_elements)
                else:
                    print(f"WARNING: Did not find three elements for protein name. Original string:")
                    print(original_line)
                    missing_three_elements += 1
                
                # Check for GN= and add "Cont_"
                gn_found = False
                for i in range(1, len(parts)):
                    if parts[i].startswith("GN="):
                        gn_found = True
                        gn_val = parts[i][3:]
                        if not gn_val.startswith("Cont_"):
                            parts[i] = "GN=Cont_" + gn_val
                        break
                        
                if not gn_found:
                    # When no GN= is found, use the third element of the protein name as GN
                    print(f"WARNING: Did not find GN=. Using third element of protein name as GN. Original string:\n")
                    print(original_line)
                    if len(prot_elements) >= 3:
                        gn_val = prot_elements[2]
                        # Remove Cont_ to avoid adding it twice
                        if gn_val.startswith("Cont_"):
                            gn_val = gn_val[5:]
                        parts.append(f"GN=Cont_{gn_val}")
                        print(f"Added GN=Cont_{gn_val}")
                    else:
                        print(f"WARNING: Did not find GN= and no third element available for GN. Original string:")
                        print(original_line)
                        missing_gn += 1
                        
                new_line = " ".join(parts) + "\n"
                f_out.write(new_line)
            else:
                f_out.write(line)
                
    print(f"Done. Processed file saved to {output_file}")
    if missing_three_elements > 0 or missing_gn > 0:
        print(f"Summary of warnings:")
        print(f"  Missing 3 elements in protein name: {missing_three_elements}")
        print(f"  Missing GN=: {missing_gn}")

if __name__ == '__main__':
    base_dir = "./"
    # The fasta file is from https://github.com/HaoGroup-ProtContLib/Protein-Contaminant-Libraries-for-DDA-and-DIA-Proteomics
    in_file = os.path.join(base_dir, "0602_Universal_Contaminants.fasta")
    out_file = os.path.join(base_dir, "0602_Universal_Contaminants_full_tags.fasta")
    
    if len(sys.argv) > 1:
        in_file = sys.argv[1]
    if len(sys.argv) > 2:
        out_file = sys.argv[2]
        
    process_fasta(in_file, out_file)
