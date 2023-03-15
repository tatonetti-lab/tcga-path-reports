#Final Report Clean-up
#Remove Non-clinically relevant (and potentially confounding) keywords from lines; Remove redaction artifacts and empty lines

import glob, os
import pickle
import pandas as pd
import shutil
import re
from collections import Counter
import matplotlib.pyplot as plt 
import seaborn as sns

#Create new directory for output

prev_dir = 'post_table_and_handwriting_removal/'
new_dir = 'post_keyword_and_empty_filtering/' 
if not os.path.exists(new_dir):
	os.makedirs(new_dir)

#Data to be processed 
current_files = glob.glob(prev_dir+'*.p')
print('Number of pages to process:',len(current_files))

#List of terms to match and remove 
list_to_remove = ['\\*',
'^end[ ]*of[ ]*report$', #footer
'^\(end[ ]*of[ ]*report\)$', #footer
'page[ ]*(\S+)[ ]*of[ ]*(\S+)', #footer
'continued[ ]*on[ ]*next[ ]*page.*', #footer
'Translated[ ]*by.*', #footer
'^Edited by.*', #footer
'.*CONTACT[ ]*YOUR[ ]*DOCTOR[ ]*WITH[ ]*THIS[ ]*REPORT.*', #footer
'.*MM/DD/YYYY.*', #remove entire line containing this expression 
'.*Ref[\. ]*Number.*', #header 
'.*DOB.*', #header 
'^D\.O\.B\..*', #header 
'^B.?Day.*',#header 
'.*Internal[ ]*Sample[ ]*ID.*', #header
'^Surg[ ]*Path[ ]*Final[ ]*Report.*$',
'^Surg\.[ ]*Path\.[ ]*No[ ]*.?:?#?$', 
'^Surg[ ]*Path.?:?#?$', 
'^Surgical[ ]*Pathology[ ]*.?:?#?$', 
'^Surgical[ ]*Pathology[ ]*Report[ ]*.?:?#?$',
'^Surgical[ ]*Pathology[ ]*Final[ ]*.?:?#?[ ]*Report[ ]*.?:?#?$',
'^Surgical[ ]*Pathology[ ]*Consultation[ ]*Report[ ]*.?:?#?$',
'^Surgical[ ]*Pathology[ ]*.?:?#?[ ]*\[OrderID[ ]*.?:?#?$',  
'^Surgical[ ]*Date[ ]*.?:?#?$',
'^Sample[ ]*.?:?#?.?:?#?$',
'^Sample[ ]*Number[ ]*.?:?#?$',
'^Sample[ ]*ID[ ]*.?:?#?$', 
'^Sample Procurement:?[ ]*Date.*',
'^TCGA.*', #TCGA barcode remnants; accounting for variations encountered due to OCR mis-translation
'Redacted.*', 
'^UUID.*',  
'^JUID.*',
'^UID:.*',
'^JID:.*',
'^IID:.*', 
'^ICGA.*',
'^ID[ ]*.?:?#?$',
'^D:\S+\-?\S+\-?\S+\-?\S+\-?\S+$',
'^D[ ]*:[ ]*\d+$',
'^ID[ ]*:[ ]*\S+\-?\S+\-?\S+\-?\S+\-?\S+$',
'^TCSA.*',
'^CGA.*',  
'.*PATH\.[ ]*NO.?:?#?.*', #header 
'^TSS.*', #header
'^[ ]*Tissue[ ]*Source[ ]*Site[ ]*.?:?#?', #header 
'^Case[ ]*#.?.*', 
'^Case.?:?#?[ ]*$', 
'^Age \\(years\\).*', 
'^Age.?:?$', #header
'.*Age/Sex.*', #header
'^Sex/Race.*$',#header
'^Tumor[ ]*ID.?:?#?$', #header
'^Date[ ]*.?:?#?$', #header
'^Date[ ]*Ordered[ ]*.?:?#?$',
'^Date[ ]*Signed[ ]*.?:?#?$',
'^Compliance[ ]*validated[ ]*by[ ]*.?:?#?$',
'^Collect[ ]*date.?:?#?',#header
'^Date[ ]*of[ ]*Procedure.?:?#?', #header 
'.*Date[ ]*of[ ]*procurement[ ]*.?:?#?', #header
'.*Date[ ]*of[ ]*tumor procurement.*',#header 
'^Date[ ]*of[ ]*report.*', #header
'^Date[ ]*of[ ]*Surgery.*', #header
'^Date[ ]*Reported.*',#header
'^Date[ ]*of.?[ ]*$',
'^Procurement[ ]*Date[ ]*.?:?#?',  
'^Date[ ]*of[ ]*Receipt.*',#header
'.*Specimen[ ]*Date/Time[ ]*.?',#header
'^Unique[ ]*Patient[ ]*Identifier.*', #header
'MRN:.*',#header
'<Sign[ ]*Out[ ]*Dr\.[ ]*Signature>.*',
'^Accession[ ]*.?:?#?[ ]*$', #header
'^Acc#.*', #header
'^PHYSICIAN[ ]*&[ ]*PAGER[ ]*#.*',#header
'^Examination[ ]*No[ ]*\..?:?#?', #header
'^Patient[ ]*Name[ ]*.?:?#?[ ]*$', #header
'^Patient\'s[ ]*Name[ ]*.?:?#?[ ]*$', #header
'^Patient[ ]*Identification[ ]*.?:?#?$', #header 
'^Patient[ ]*Results[ ]*:?#?[ ]*$', #header 
'^Procurement[ ]*.?:?#?$$',#header 
'Name[ ]*of[ ]*Pathologist.*',#header
'^Assistant[ ]*.?:?#?$', #header
'^Pathology[ ]*Assistant[ ]*.?:?#?$',
'^Pathologists\'[ ]*Assistant[ ]*.?:?#?$',
'^Performed[ ]*by[ ]*Pathologist\'s[ ]*Assistant[ ]*.?:?#?$',
'.*MED\.[ ]*REC\.[ ]*N.*', #header 
'.*Med\.[ ]*Rec\.[ ]*#.*', #header
'^Admit[ ]*Protocol[ ]*.?:?#?$',
'\\(\\)', #empty parentheses
'\[[ ]*\]',
'Printed[ ]*by[ ]*.?:?#?$', #footer
'Printed[ ]*on[ ]*.?:?#?$', #footer
'^Verified[ ]*By.*',#footer
'Report[ ]*Electronically[ ]*Signed.*',#footer          
'^History[ ]*Case[ ]*Pathology.*', #header
'^File[ ]*under.*', #header
'^Admitting Physician.*', 
'^Result[ ]*Type.?:?#?$', #header
'^Result[ ]*Title.*', #header
'^Result[ ]*Status.?:?#?$', #header
'Patient:[ ]*XXX', #header 
'^Examination performed on.*', #header
'^Internal[ ]*invoice.*', #header 
'PESEL:[ ]*XXX.*',
'^OPER[ ]*.?:?#?$',
'^Service.?:?#?$',
'^Ref.?[ ]*Physician.*',
'^Account[ ]*#.?$',
'^OUTPATIENT.?:?#?$',
'^Patient[ ]*Address.*', 
'^Billing.*$',
'.*Additional Copy to.*',
'^Ref\.[ ]*Source.*',
'^INPATIENT.?:?#?$',
'^Other[ ]*Related[ ]*Data.?:?#?$',
'^Financial[ ]*Number.*',
'^Performing[ ]*Clinician.?:?#?$',
'^Street[ ]*address.*',
'^Internal[ ]*postal[ ]*address.*',
'^Patient[ ]*Key.*$',
'^E\-mail.*$',
'^Concerning.?:?#?$',
'^Direct[ ]*dial[ ]*.?:?#?$',
'^Path.?\-[ ]*No.?.?[ ]*$',
'^Mobile[ ]*.?:?#?$', 
'^Postal[ ]*address[ ]*.?:?#?$',
'^Pt\.[ ]*Phone[ ]*no[ ]*.?:?#?$',
'^PATIENT[ ]*PHONE[ ]*NO[ ]*.?:?#?$',
'^HCN[ ]*.?:?#?$',
'.*AP[ ]*Report[ ]*.?:?#?$',
'^Final[ ]*Report[ ]*.?:?#?$',
'^Document[ ]*Type[ ]*.?:?#?$',
'^Document[ ]*Status[ ]*.?:?#?$',
'^Document[ ]*.?:?#?$', 
'^Auth[ ]*\\(Verified\\)',
'^Document[ ]*Title[ ]*.?:?#?$',
'^Anatomic[ ]*Pathology[ ]*Report[ ]*.?:?#?$',
'^Encounter[ ]*info[ ]*.?:?#?$',
'^Value[ ]*of[ ]*diagnostic[ ]*procedure[ ]*.?:?#?$', 
'^Result[ ]*.?:?#?$',
'^Received?[ ]*.?.?:?#?$' ,
'^Entry[ ]*Information[ ]*.?:?#?$',
'^Seen[ ]*in[ ]*consultation[ ]*with[ ]*.?:?#?$',
'^continued[ ]*next[ ]*.?:?#?$',
'^Entry[ ]*Date[ ]*and[ ]*Time.?[ ]*$',
'^Lab[ ]*Status[ ]*.?:?#?$',
'^Entered[ ]*by[ ]*.?:?#?$',
'^Final[ ]*result[ ]*.?:?#?$',
'^Reported[ ]*by[ ]*.?:?#?$',
'^Report[ ]*.?:?#?$',
'^Physician[ ]*.?:?#?$',
'^Collected[ ]*.?:?#?$',
'^Specimen[ ]*Received*.?:?#?$',
'^Pathology[ ]*Number[ ]*.?:?#?$',
'^Inquiry[ ]*Number.?[ ]*.?:?#?$',
'^Received[ ]*Time[ ]*.?:?#?$',
'^Received[ ]*Date[ ]*.?:?#?$',              
'^Time[ ]*Collected[ ]*.?:?#?$',
'^Time[ ]*Received[ ]*.?:?#?$',
'^Time[ ]*Reported[ ]*.?:?#?$',
'^Time[ ]*Transmitted[ ]*.?:?#?$',
'^Collected[ ]*Time[ ]*.?:?#?$',
'^CC:$',
'^Final[ ]*.?:?#?$',
'^Relevant Information[ ]*.?:?#?$',
'^Location[ ]*.?:?#?$',
'^Copied[ ]*To[ ]*.?:?#?$',
'^Report[ ]*Patient[ ]*.?:?#?$',
'^Name[ ]*.?:?#?$',
'^Accession[ ]*Number[ ]*.?:?#?$',
'^Reviewed[ ]*.?:?#?$',
'^Reviewed[ ]*and[ ]*electronically.*',
'^Reviewed[ ]*.?:?#?$',
'^Reviewed[ ]*by .*',
'^Reviewed[ ]*By[ ]*:?[ ]*List.*',
'^Grossed[ ]*by[ ]*:?.*',
'^Reviewed[ ]*by[ ]*dr.*',
'^Lab[ ]*Information[ ]*.?:?#?$',
'^Parent[ ]*Order[ ]*.?:?#?$',
'^Child[ ]*Order[ ]*.?:?#?$',
'^Result[ ]*Information[ ]*.?:?#?$',
'^Result[ ]*and Time[ ]*.?:?#?$',
'^Edited[ ]*.?:?#?$',
'^Entry[ ]*Date[ ]*.?:?#?$',
'^Component[ ]*Results[ ]*.?:?#?$',
'^Taken[ ]*.?:?#?$',
'^Physician\\(s\\)[ ]*.?:?#?$',
'^Reported[ ]*.?:?#?$',
'^Reported[ ]*Date[ ]*.?:?#?$',
'^BIRTH[ ]*.?:?#?$',
'^PAT[ ]*TYPE[ ]*.?:?#?$',
'^ADM[ ]*.?:?#?[ ]*.?:?#?$',
'^Unit[ ]*in[ ]*charge[ ]*.?:?#?$',
'^Physician[ ]*in[ ]*charge[ ]*.?:?#?$',
'^Material[ ]*collected[ ]*on[ ]*.?:?#?$',
'^Material[ ]*received[ ]*on[ ]*.?:?#?$',
'^Address[ ]*.?:?#?$',
'^MR[ ]*No[ ]*.?:?#?$',
'^Taken/Received[ ]*.?:?#?$',
'^Service[ ]*.?:?#?$',
'^Surgeon/physician[ ]*.?:?#?$',
'^Attending[ ]*Surgeon[ ]*.?:?#?$',
'^Surgeons?[ ]*.?.?:?#?$',
'^Gender[ ]*.?.?:?#?$',
'^Sex[ ]*.?.?:?#?$',
'^Case[ ]*Number[ ]*.?:?#?$',
'^Phone[ ]*.?.?:?#?$',
'^Fax[ ]*.?.?:?#?$',
'^\\(Continued\\)[ ]*.?:?#?$',
'^Tel[ ]*.?:?#?$',
'^Surname[ ]*.?:?#?$',
'^Surname[ ]*and[ ]*Name.?:?#?$',
'^Forename.?\\(s\\)[ ]*.?:?#?$',
'^Unit[ ]*.?:?#?$',
'^Request Date[ ]*.?:?#?$',
'^This[ ]*Copy[ ]*For[ ]*.?:?#?$',
'^Path#[ ]*.?:?#?$',
'^Staff[ ]*Pathologist[ ]*.?:?#?$',
'^Other[ ]*Pathologists[ ]*/[ ]*PAs[ ]*.?:?#?$',
'^Dictated[ ]*by[ ]*.?:?#?$',
'^MR[ ]*.?:?#?$',
'^Signature[ ]*.?:?#?$',
'^COMMITTEE.*',
'^Requested[ ]*by[ ]*.?:?#?$',
'^AP[ ]*Surgical[ ]*Pathology[ ]*.?:?#?$',
'^Performed[ ]*by[ ]*.?:?#?$',
'^Performed[ ]*by[ ]*the[ ]*staff[ ]*pathologist[ ]*.?:?#?$',
'^MED.?[ ]*REC[ ]*.?:?#?$',
'PATIENT[ ]*PHONE[ ]*.?:?#?$', 
'^CONF[ ]*.?:?#?$',
'^Collection.*', 
'^Print[ ]*time.*',
'^professor[ ]*of[ ]*pathology[ ]*.?:?#?$',
'Filed[ ]*automatically[ ]*on.*',
'^by[ ]*.?:?#?$',
'Report[ ]*electroni.*',
'^Electronic.*',
'Excision[ ]*date.*',
'Formatted[ ]*Path[ ]*Reports?[ ]*.?:?#?$',
'Compliance[ ]*vali[ ]*by.*',
'^Status[ ]*:[ ]*.?:?#?complete$',
'^Status[ ]*.?:?#?$',
'^Status[ ]*History[ ]*.?:?#?$',
'^Status[ ]*.?:?#?[ ]*Signed.*$',
'^Status[ ]*.?:?#?[ ]*Corrected$',             
'^Status[ ]*:[ ]*.?:?#?supplemental$',
'^STATUS[ ]*:[ ]*SOUT[ ]*.?:?#?$',
'^Status[ ]*:[ ]*N/A[ ]*.?:?#?$',
'^Status[ ]*:[ ]*simed[ ]*Out[ ]*.?:?#?$',
'^Status[ ]*:[ ]*Ordered[ ]*.?:?#?$',
'^Status[ ]*Final[ ]*result[ ]*.?:?#?$',
'^Status:[ ]*Siman[ ]*Out[ ]*.?:?#?$',
'^Status[ ]*\\(Last[ ]*updated[ ]*Date/Time\\)[ ]*.?:?#?$',
'^Status:[ ]*Complete[ ]*as[ ]*of[ ]*.?:?#?$',
'^Status:[ ]*Final[ ]*resuit[ ]*.?:?#?$',
'^Status:[ ]*Final[ ]*result[ ]*.?:?#?$',
'^Status[ ]*of[ ]*Proceduze:[ ]*Signed[ ]*Out[ ]*.?:?#?$',
'^Status[ ]*:[ ]*DIS[ ]*IN[ ]*.?:?#?$',       
'^Department[ ]*of[ ]*Cancer Pathology.*',
'^Department[ ]*of[ ]*Pathology[ ]*.?:?#?$',
'Submitting[ ]*Physician.*',
'^Resulting[ ]*Agency[ ]*.?:?#?$',
'^Provider[ ]*Information[ ]*.?:?#?$',
'Accessioned[ ]*on.*',
'CASE NO[ ]*.?:?#?$',
'^OpenClinica.*',
'^Order.*', 
'^Test[ ]*performed[ ]*at.*',
'Addendum[ ]*Report[ ]*Issued.*',
'^MRN[ ]*.?:?#?[ ]*.?:?#?[ ]*.?:?#$',
'This[ ]*report[ ]*has[ ]*been[ ]*finalized.*',
'^FMP/SSN.*',
'^SSN.*',
'^Referring[ ]*Physician[ ]*.?:?#?$',
'^Subm[ ]*Dr[ ]*.?:?#?.?:?#?$',
'^Acct[ ]*.?:?#?$',
'^Reg[ ]*.?:?#?$',
'^Disch[ ]*.?:?#?$',
'^Att[ ]*Dr.*',
'^Req[ ]*.?:?#?$',
'^Ref[ ]*.?:?#?$',
'^Hosp[ ]*.?:?#?$',
'^Prov[ ]*.?:?#?$',
'^\\(FINAL\\)[ ]*.?:?#?$',
'^Patient[ ]*Na[ ]*.?:?#?$',
'Visit[ ]*.?:?#?$',
'^Path[ ]*.?:?#?$',
'^Primary[ ]*Path[ ]*.?:?#?$',                             
'^Resident[ ]*Path[ ]*.?:?#?$',
'^Copath[ ]*[ ]*.?:?#?$',
'^Department[ ]*:[ ]*PATH[ ]*.?:?#?$',
'^Formatted[ ]*Path[ ]*.?:?#?$',                  
'Copy[ ]*To.*',
'^Transcribed[ ]*by.*',
'^Surgery[ ]*.?:?#?$',
'^XX[ ]*.?:?#?$', 
'^XXX+[ ]*.?:?#?$',
'^Signed[ ]*.?:?#?$',
'^Signed[ ]*out[ ]*.?:?#?$',
'^Signed[ ]*by[ ]*.?:?#?$',
'^Signed[ ]*out[ ]*by[ ]*.?:?#?$',
'^\[][ ]*.?:?#?$',
'Surg[ ]*Date.*',
'^TBCF#[ ]*.?:?#?$:',
'^Gross[ ]*Dictation.*', 
'^Unique[ ]*Patient[ ]*.?:?#?$',
'^Hospital[ ]*.?:?#?$',
'^Patient[ ]*Type[ ]*.?:?#?$', 
'^Physician[ ]*Copy[ ]*.?:?#?$',
'^PATHOLOGY[ ]*REPORT[ ]*.?:?#?$',
'^Component[ ]*Results[ ]*.?:?#?$',
'Patient[ ]*Number[ ]*.?:?#?$', 
'Room/Bed.*', 
'Ethnicity \\(Race\\)[ ]*.?:?#?$',
'^Cancer[ ]*Sample[ ]*.?:?#?$',  
'^Page[ ]*.?:?#?$',
'^Page:[ ]*(S+)$',
'^Pathologist[ ]*.?:?#?$', 
'^Patient[ ]*.?:?#?$', 
'^PAT[ ]*TYPE[ ]*.?:?#?$',
'.*1974.*',#PRIVACY ACT OF 1974
'^No[ ]*.?:?#?$',
'^Yes[ ]*.?:?#?$', 
'^\.+$',
'^,+$',
'^:+$', 
'^-+$',        
'^Collect[ ]*.?:?#?$',
'^[:\.,\)\(\?\*;]?[ ]*[:\.,\)\(\?\*;-]?[ ]*[:\.,\)\(\*;]?[ ]*[:\.,\)\(\?\*;-]?[ ]*[:\.,\)\(\*;-]?$'] #Matches if a line contains only punctuation (with any number of spaces in between)


#List of terms that require capitalization for specific-matching 
full_cap_list=['.*DOB.*','^D\.O\.B\..*',  
'^DOB.*',
'^UUID.*', 
'^JUID.*',
'^UID:.*',
'^JID:.*',
'^ICGA.*',
'^ID[ ]*.?:?#?$',
'^TCSA.*',
'^CGA.*',  
'^TCGA.*',
'^HCN[ ]*.?:?#?$',
'^TSS.*']


#List of terms that require specifically the first letter to be capitalized for matching 
capitalize_first_letter = ['^Case[ ]*#.?.*', 
'^Case.?:?#?[ ]*$', 
'^Patient[ ]*.?:?#?$', 
'^Patient[ ]*Name[ ]*.?:?#?[ ]*$', #header
'^Patient\'s[ ]*Name[ ]*.?:?#?[ ]*$',
'^Patient[ ]*Identification[ ]*.?:?#?$', #header 
'^Patient[ ]*Results[ ]*:?#?[ ]*$', #header 
'PATIENT[ ]*PHONE[ ]*.?:?#?$', 
'^PATIENT[ ]*PHONE[ ]*NO[ ]*.?:?#?$',
'Patient:[ ]*XXX', #header 
'^Patient[ ]*Na[ ]*.?:?#?$',
'^Patient[ ]*Address.*', 
'^Patient[ ]*Key.*$',
'^Patient[ ]*Type[ ]*.?:?#?$',
'Patient[ ]*Number[ ]*.?:?#?$', 
'^Sample[ ]*.?:?#?.?:?#?$',
'^Sample[ ]*Number[ ]*.?:?#?$',
'^Sample[ ]*ID[ ]*.?:?#?$', 
'^Sample Procurement:?[ ]*Date.*',
'^Taken[ ]*.?:?#?$',
'^Reported[ ]*.?:?#?$',
'^Reported[ ]*by[ ]*.?:?#?$',
'^Reported[ ]*Date[ ]*.?:?#?$',
'^Order.*', 
'^Collection.*', 
'^Procurement[ ]*.?:?#?$',
'^Procurement[ ]*Date[ ]*.?:?#?$', 
'^Status[ ]*:[ ]*.?:?#?complete$',
'^Status[ ]*.?:?#?$',
'^Status[ ]*History[ ]*.?:?#?$',
'^Status[ ]*.?:?#?[ ]*Signed.*$',
'^Status[ ]*.?:?#?[ ]*Corrected$',             
'^Status[ ]*:[ ]*.?:?#?supplemental$',
'^STATUS[ ]*:[ ]*SOUT[ ]*.?:?#?$',
'^Status[ ]*:[ ]*N/A[ ]*.?:?#?$',
'^Status[ ]*:[ ]*simed[ ]*Out[ ]*.?:?#?$',
'^Status[ ]*:[ ]*Ordered[ ]*.?:?#?$',
'^Status[ ]*Final[ ]*result[ ]*.?:?#?$',
'^Status:[ ]*Siman[ ]*Out[ ]*.?:?#?$',
'^Status[ ]*\\(Last[ ]*updated[ ]*Date/Time\\)[ ]*.?:?#?$',
'^Status:[ ]*Complete[ ]*as[ ]*of[ ]*.?:?#?$',
'^Status:[ ]*Final[ ]*resuit[ ]*.?:?#?$',
'^Status:[ ]*Final[ ]*result[ ]*.?:?#?$',
'^Status[ ]*of[ ]*Proceduze:[ ]*Signed[ ]*Out[ ]*.?:?#?$',
'^Status[ ]*:[ ]*DIS[ ]*IN[ ]*.?:?#?$',             
'^Reviewed[ ]*.?:?#?$',
'^Reviewed[ ]*and[ ]*electronically.*',
'^Reviewed[ ]*.?:?#?$',
'^Reviewed[ ]*by .*',
'^Reviewed[ ]*By[ ]*:?[ ]*List.*',
'^Surgeon/physician[ ]*.?:?#?$',
'^Surgeons?[ ]*.?.?:?#?$',
'^Pathologist[ ]*.?:?#?$', 
'^Surgical[ ]*Pathology[ ]*.?:?#?$', 
'^Surgical[ ]*Pathology[ ]*Report[ ]*.?:?#?$',
'^Surgical[ ]*Pathology[ ]*Final[ ]*.?:?#?[ ]*Report[ ]*.?:?#?$',
'^Surgical[ ]*Pathology[ ]*Consultation[ ]*Report[ ]*.?:?#?$',
'^Surgical[ ]*Pathology[ ]*.?:?#?[ ]*\[OrderID[ ]*.?:?#?$',  
'^Surgical[ ]*Date[ ]*.?:?#?$']

#Define function for line removal, specifying capitalization conditions for keywords as outlined above 
def process_line(new_line, list_to_remove, keyword_lines_removed_dict):
        
    for element in list_to_remove:

        old_line = new_line
        prev = len(old_line)

        if element in full_cap_list:
            if element == '.*DOB.*':
                if 'ENDOBRONCHIAL' not in new_line:
                    new_line = re.sub(element, "", new_line).strip()
            else:
                new_line = re.sub(element, "", new_line).strip()

        elif element in capitalize_first_letter:
            if len(new_line) > 0: 
                if new_line[0].isupper():
                    new_line = re.sub(element, "", new_line, flags=re.IGNORECASE).strip()

        else:
            if element in ['^No[ ]*.?:?#?$','^Yes[ ]*.?:?#?$']:
                if (i != 0) and lines[i-1].strip()[-1]==':': #Ammending yes/no removal
                    pass
                else:
                    new_line = re.sub(element, "", new_line, flags=re.IGNORECASE).strip()
            new_line = re.sub(element, "", new_line, flags=re.IGNORECASE).strip()

        if len(new_line) != prev:
            keyword_lines_removed_dict[element].append(old_line)
            
    return new_line, keyword_lines_removed_dict

#Track lines removed (and which keywords trigger removal)
keyword_lines_removed_dict = {elem:[] for elem in list_to_remove}
n_lines_removed_dict = {}

#Track number of empty pages (removed)
n_empty_pages = 0

#Note: The below code may take some time to run 
for j in range(len(current_files)):
    
    p = current_files[j]
    n_lines_removed = 0
    new_lines = []
    lines = pickle.load(open(p, "rb"))
    
    for i in range(len(lines)):
        
        len_line = len(lines[i])
        new_line = lines[i].strip() #Remove leading and trailing whitespace
        new_line = re.sub(' {2,}', ' ', new_line) #Remove >1 white space between words
        
        #3x keyword removal
        for cycle in range(3):
            new_line, keyword_lines_removed_dict = process_line(new_line, list_to_remove, keyword_lines_removed_dict)

        
        if (len_line != 0) and (len(new_line)==0):
            n_lines_removed +=1 
        #Remove lines that were originally empty or whitespace, or that are now whitespace
        elif (len_line != 0) and (len(lines[i].strip())!=0) and (len(new_line.strip())!= 0): 
            new_lines.append(new_line)
    
    #Only save non-empty pages
    if len(new_lines) > 0:
        pickle.dump(new_lines, open(new_dir + p.replace(prev_dir,""),"wb"))
    else:
        n_empty_pages +=1 
    n_lines_removed_dict[p.replace(prev_dir,'').replace('_lines.p','')] = n_lines_removed

    #Track progress
    if j % 2000 == 0: 
        print(j,' pages finished.')

print(n_empty_pages,' empty pages, removed.')
