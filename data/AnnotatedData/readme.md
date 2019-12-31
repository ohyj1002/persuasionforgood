### Description of CSV fields

1. 300_info.csv
    B2: Dialogue ID
    B3: User ID  
    B4: Role (0 means persuader, 1 means persuadee)  
    B5: Intended donation mentioned in dialogue (annotated by human, for the persuadee side only)
    B6: Actual donation made by the participants after the session ended
    B7: number of Turns 
2. dialog_310.csv
    * B2: Dialogue ID
    * B4: Role (0 means persuader, 1 means persuadee) 
    * Turn: Turn index 
    * Unit: Sentence in utterances
    * er_label_1: The most salient label for the persuader side
    * ee_label_1: The most salient label for the persuadee side
    * er_label_2: The 2nd most salient label for the persuader side, most utterances don't have a second label
    * er_label_2: The 2nd most salient label for the persuadee side, most utterances don't have a second label
    * neg,neu,pos : Three sentiment values
   
    

