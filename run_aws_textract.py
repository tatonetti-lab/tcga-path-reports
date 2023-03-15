#Run pathology report images through AWS Textract (on server)
import random
import timeit
import glob, os
import pickle
import boto3

#Make directory for AWS Response files
new_directory = 'data/aws_response/'
if not os.path.exists(new_directory):
    os.makedirs(new_directory)

#Load Pathology Report Images - Separate image for each report page
imgs = glob.glob('imgs_for_aws/*.jpg') 

#Initialize client 
client = boto3.client('textract')

#Run AWS Textract
start = timeit.default_timer()

i=0 #Track Number of Reports Processed

for img in imgs: 

    i+=1

    #Convert Image to bytearray
    with open(img, "rb") as image: #File will close automatically
        f = image.read()
        b = bytearray(f)

    #Run Analyze_Document, with table-annotation 
    response_with_table = client.analyze_document(
    Document={'Bytes': b}, #base64-encoded bytes
    FeatureTypes=['TABLES']) 
    
    #Save AWS Response
    pickle.dump(response_with_table, open(new_directory + img.split('/')[-1].replace('.jpg','_response')+".p", "wb" ))

    if i%100 == 0:
        print('Number of pages processed: ', i, ' Runtime (min):', round((timeit.default_timer()-start)/60,2))

stop = timeit.default_timer()

print('Total runtime: ', round((stop - start)/60/60,2), 'hours') 