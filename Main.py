import glob
import errno
import spacy
import json
import os
import nltk
import neuralcoref
import copy
import string
from nltk.corpus import wordnet as wn
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from itertools import chain
from nltk.stem import PorterStemmer
from nltk import Tree
from nltk.parse.stanford import StanfordParser
from nltk.corpus import stopwords
from spacy import displacy
from spacy.matcher import Matcher
from nltk.tokenize import sent_tokenize, word_tokenize
from spacy.pipeline import EntityRuler
from spacy.tokens import Doc
nlp = spacy.load('en_core_web_sm')
ps = PorterStemmer()
check_list=["part","belong","in","citi","insid","capit","state","orgin","is","locat"]
job_list=['executive', 'actress', 'host', 'producer', 'philanthropist', 'queen', 'barber', 'president', 'miner', 'city councilman', 'farmer', 'preacher', 'maid', 'student', 'news anchor', 'critic', 'columnist', 'candidate', 'author', 'housewife', 'judge', 'princess', 'personal trainer', 'reader', 'model student', 'journalist', 'biographer', 'reporter', 'king', 'filmmaker', 'editor', 'therapist', 'entertainer', 'ceo', 'senator', 'chairman', 'politician', 'leader', 'pope', 'springer', 'professor', 'attorney', 'governor', 'crown prince', 'teacher', 'premier', 'mayor', 'magician', 'executive producer', 'magnate', 'vice president', 'founder', 'congressman', 'stockbroker', 'salesman', 'analyst', 'general', 'janitor', 'boss', 'doctor', 'activist', 'owner', 'director', 'trader', 'chief financial officer', 'publisher', 'companion', 'assistant coach', 'manager', 'mediator', 
          'secretary of the treasury', 'actuary', 'manufacturer', 'river', 'surveyor', 'lieutenant governor', 'commander', 'envoy', 'lieutenant colonel', 'translator', 'captain', 'colonel', 'commander colonel', 'brigadier general', 'major general', 'guard', 'soldier', 'secretary', 'baron', 'chief of staff', 'major', 'admiral', 'president general', 'coach', 'chancellor', 'administrator', 'merchant', 'attorney general', 'secretary of state', 'secretary of war', 'diplomat', 'chief justice', 'negotiator', 'minister', 'principal', 'secretary of treasury', 'historian', 'lieutenant general', 'speaker', 'reverend', 'architect', 'dentist', 'dancer', 'pastor', 'creator', 'charter', 'entrepreneur', 'engineer', 'designer', 'co-founder', 'co-chairman', 'model', 'pilot', 'sailor', 'commodore', 'guide', 'chief executive officer', 'chief technology officer', 'astronaut', 'scientist', 
          'gen.', 'geographer', 'emperor', 'theologian', 'printer manufacturer', 'recorder', 'general manager', 'salesmen', 'vendor', 'graphic designer', 'inventor', 'secretary of housing and urban development', 'secretary of transportation', 'referee', 'dealer', 'driver', 'collector', 'vice-president', 'demonstrator', 'cell maker', 'private', 'spokesman', 'buyer', 'cfo', 'managing director', 'chief executive', 'retailer', 'printer', 'developer', 'processor', 'grip', 'chief operating officer', 'assistant', 'layer', 'operator', 'header', 'writer', 'singer', 'evangelist', 'executive director', 'general counsel', 'city manager', 'physician', 'importer', 'explorer', 'empress', 'boxer', 'general secretary', 'party leader', 'representative', 'secretary of defense', 'prince', 'director-general', 'fund manager', 'surgeon', 'cook', 'comptroller', 'refiner', 'tanker', 'vice-chairman', 
          'executive chairman', 'constable', 'interim president', 'nobel laureate', 'dean', 'artist', 'landscape architect', 'consultant', 'chef', 'vice chairman', 'superior', 'jeweler', 'specialist', 'broker', 'strategist', 'treasury secretary', 'underwriter', 'quality control supervisor', 'auditor', 'spokeswoman', 'district attorney', 'principal author', 'treasurer', 'lobbyist', 'deputy mayor', 'communications director', 'assistant attorney general', 'executive vice president', 'chief compliance officer', 'lawyer', 'spokesperson', 'technician', 'intelligence director', 'hacker', 'astronomer', 'composer', 'aerospace engineer', 'homemaker', 'marketing manager', 'businesswoman', 'monk', 'explorer captain', 'builder', 'state treasurer', 'superintendent', 'governor general', 'prime minister', 'chief minister', 'poet', 'novelist', 'indian activist', 'clerk', 'barrister', 'priest', 
          'landlady', 'magistrate', 'police officer', 'saint', 'dictator', 'representative leader', 'governor-general', 'marshal', 'philosopher', 'butcher', 'missionary', 'sultan', 'interpreter', 'economist', 'physicist', 'musician', 'custodian', 'investment banker', 'financier', 'secretary of commerce', 'secretary of labor', 'performer', 'legislator', 'actor', 'cabinetmaker', 'carpenter', 'servant', 'ambassador', 'chief of staff general', 'rep.', 'campaign manager', 'jurist', 'whig activist', 'orderly', 'sociologist', 'bishop', 'botanist', 'sheriff', 'chief of police', 'firefighter', 'cartographer', 'lt. col.', 'anthropologist', 'minority leader', 'food critic', 'playwright', 'cowboy', 'first lady', 'agriculture commissioner', 'corporal', 'flyer', 'software engineer', 'navigator', 'businessman', 'steward', 'comedian', 'grocer', 'student activist', 'machinist', 'hatter', 'babysitter', 
          'waitress', 'computer scientist', 'tipper', 'hockey player', 'researcher', 'broadcaster', 'thinner']
ruler = EntityRuler(nlp)
patterns = [{"label": "BUY", "pattern": "purchased"}, {"label": "BUY", "pattern": "purchased by"},
                {"label": "BUY", "pattern": "acquired by"}, {"label": "BUY", "pattern": "acquired"},
            {"label": "BUY", "pattern": "acquire"}, {"label": "BUY", "pattern": "bought"}, 
            {"label": "BUY", "pattern": "bought  by"},{"label": "BUY", "pattern": "took over"},
           {"label": "BUY", "pattern": "owns"},{"label": "BUY", "pattern": "owned"},
            {"label": "BUY", "pattern": "own"}]
ruler.add_patterns(patterns)
nlp.add_pipe(ruler)
merge_nps = nlp.create_pipe("merge_noun_chunks")
nlp.add_pipe(merge_nps)
neuralcoref.add_to_pipe(nlp)

def read_single_file_with_coref(filename):
    sent_tokens = []
    f = open(filename,encoding="ascii",errors="ignore")
    temp=f.read()
    temp=nlp(temp)
    temp=temp._.coref_resolved
    for sent in nltk.sent_tokenize(temp):
        sent_tokens.extend(sent.split('\n\n'))
    f.close()
    return sent_tokens

def read_single_file_table(file_name):
    f = open(file_name, encoding="ascii", errors="ignore")
    lines = f.readlines()
    sentences = ''
    for line in lines:
        if line.find('    ') == 0 or '\t' in line:
            sentences = sentences + line.replace('\n','.')
    f.close()
    sent_tokens =  sent_tokenize(sentences)
    return sent_tokens

def merge_ents(doc):
    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            retokenizer.merge(ent)
    return doc

def check_job(text):
    text=text.split()
    for i in text:
        if(i.lower() in job_list):
            return True
    return False
#///////////////////////////////////////////////BUY TEMPLATE\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


def extract_buy_templates(doc):
    template = {"buyer": "", "item": "", "price": "", "quantity": "", "source": ""}
    list_of_templates = []
    
    # A BUY B for MONEY
    for head in doc:
        if(head.ent_type_=="BUY"):
            for token in head.children:
                if (token.dep_ == "nsubj"):
                    template["buyer"] = token
                if((token.pos_=="NOUN" and token.dep_ == "dobj")):
                    template["item"] = token
                    for j in token.children:
                        if(j.dep_=="nummod"):
                            template["quantity"]=j
                elif (token.dep_ == "dobj"):
                    template["item"] = token
                for i in doc:
                    if(i.ent_type_ == "MONEY"):
                        if(head in list(i.ancestors)):
                            template["price"] = i
                    if(i.text.lower()=="from" or i.text.lower()=="of" or i.text.lower()=="in"):
                        for j in i.children:
                            if(j.pos_=="PROPN" and j.ent_type_!="GPE"):
                                template["source"]=j
                if (len(template["buyer"]) > 0 and len(template["item"]) > 0):
                    list_of_templates.append(template)
                    template = {"buyer": "", "item": "", "price": "", "quantity": "", "source": ""}
    
    # B was BUY by A for MONEY
    
    for head in doc:
        if(head.ent_type_=="BUY"):
            for token in head.children:
                if((token.pos_=="NOUN" and token.dep_ == "nsubjpass")):
                    template["item"] = token
                    for j in token.children:
                        if(j.dep_=="nummod"):
                            template["quantity"]=j
                elif ( token.dep_ == "nsubjpass"):
                    template["item"] = token
                if ( token.dep_ == "pobj" ):
                    template["buyer"] = token
                
                for i in doc:
                    if(i.ent_type_ == "MONEY"):
                        if(head in list(i.ancestors)):
                            template["price"] = i
                    if(i.text.lower()=="from" or i.text.lower()=="of" or i.text.lower()=="in"):
                        for j in i.children:
                            if(j.pos_=="PROPN" and j.ent_type_!="GPE"):
                                template["source"]=j
                if (len(template["buyer"]) > 0 and len(template["item"]) > 0):
                    list_of_templates.append(template)
                    template = {"buyer": "", "item": "", "price": "", "quantity": "", "source": ""}
    
    return list_of_templates

#///////////////////////////////////////////////PART TEMPLATE\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# PARSE TREE APPROCH 

def place_parse_tree(text):
    doc = nlp(text)
    doc=merge_ents(doc)
    
    def check(text):
        count=0
        for token in doc:
            if(token.text==text):
                temp=list(token.ancestors)
                for i in temp:
                    if(ps.stem(i.text) in check_list):
                         count+=1
                break
        if(count>1):
            return True
        return False
    
    left_list=[]
    temp_str=""
    for sent in doc.sents:
        for child in sent.root.children:
            if(child.ent_type_ == "GPE"):
                left_list.append(child.text)
                break
    if(len(left_list)==0):
        return []

    next_ind=text.find(left_list[0])+len(left_list[0])+2
    for ent in doc.ents:
        if(ent.label_ == "GPE" and ent.text not in left_list):
            if(next_ind==ent.start_char):
                left_list.append(ent.text)
                next_ind=ent.end_char+2
            elif(next_ind>0 and next_ind+3<len(text) and (next_ind+3==ent.start_char or next_ind+4==ent.start_char) and (text[next_ind-1:next_ind+2]=="and" or text[next_ind:next_ind+3]=="and")):
                left_list.append("&")
                left_list.append(ent.text)
                next_ind=ent.end_char+2


    right_list=[]
    for ent in doc.ents:
        if(ent.label_ == "GPE" and ent.text not in left_list and check(ent.text)==True):
            right_list.append(ent.text)
            break

    if(len(right_list)==0):
        return []
    next_ind=text.find(right_list[0])+len(right_list[0])+2
    for ent in doc.ents:
        if(ent.label_ == "GPE" and ent.text not in left_list and ent.text not in right_list and check(ent.text)==True):
            if(next_ind==ent.start_char):
                right_list.append(ent.text)
                next_ind=ent.end_char+2
            elif(next_ind>0 and next_ind+3<len(text) and (next_ind+3==ent.start_char or next_ind+4==ent.start_char) and (text[next_ind-1:next_ind+2]=="and" or text[next_ind:next_ind+3]=="and")):
                right_list.append("&")
                right_list.append(ent.text)
                next_ind=ent.end_char+2;

    parse_ans=[]
    if(len(left_list)==1 and len(right_list)==1):
        temp=(left_list[0],right_list[0])
        parse_ans.append(temp)
    elif("&" in left_list and "&" in right_list):
        left_list.remove("&")
        right_list.remove("&")
        for i in left_list:
            for j in right_list:
                temp=(i,j)
                parse_ans.append(temp)
    elif("&" in left_list and "&" not in right_list):
        left_list.remove("&")
        for i in left_list:
            temp=(i,right_list[0])
            parse_ans.append(temp)
    return parse_ans

# Holonyms Approch

def place_holonym(text):
    doc = nlp(text)
    doc=merge_ents(doc)
    def holo(word):
        holonyms_list = []
        for i,j in enumerate(wn.synsets(word)):
            holonyms_list.extend(list(chain(*[l.lemma_names() for l in j.part_holonyms()])))
        return holonyms_list
    holo_ans=[]
    gpe_ents=[]
    for ent in doc.ents:
        if(ent.label_ == "GPE"):
            gpe_ents.append(ent.text)
    for i in gpe_ents:
        ans_list=holo(i.replace(" ","_"))
        for j in gpe_ents:
            if (j.replace(" ","_") in ans_list):
                anss=(i,j)
                holo_ans.append(anss)
    final_holo=[]
    for i in holo_ans:
        if(i[0]!=i[1]):
            final_holo.append(i)
    return final_holo

#  REGEX APPROCH

def place_regex(text):
    doc = nlp(text)
    doc=merge_ents(doc)
    ans=[]
    next_ind=-1
    for ent in doc.ents:
        if(ent.label_ == "GPE"):
            if(next_ind==-1 or next_ind==ent.start_char):
                ans.append(ent.text)
                next_ind=ent.end_char+2
            elif(next_ind>0 and next_ind+3<len(text) and (next_ind+3==ent.start_char or next_ind+4==ent.start_char) and (text[next_ind-1:next_ind+2]=="and" or text[next_ind:next_ind+3]=="and")):
                ans.append("&")
                ans.append(ent.text)
                next_ind=ent.end_char+2;
            else:
                ans.append("#")
                ans.append(ent.text)
                next_ind=ent.end_char+2;
    rem_list=[]
    for i in range(len(ans)):
        if(ans[i]=="&"):
            rem_list.append(i)
            rem_list.append(i+1)
            j=i-1
            while(j!=0 and ans[j+1]!="#"):
                rem_list.append(j)
                j=j-1
    rem_list = list(dict.fromkeys(rem_list))
    for index in sorted(rem_list, reverse=True):
        del ans[index]
    ans_regex=[]
    temp_tup=()
    for i in (range(len(ans)-1)):
        if(ans[i+1]=="#" or ans[i]=="#"):
            continue
        else:
            temp_tup=(ans[i],ans[i+1])
            ans_regex.append(temp_tup)
    return ans_regex

# MERGING 

def place_template(text):
    final_ans=set()
    final_ans.update(place_regex(text))
    final_ans.update(place_holonym(text))
    final_ans.update(place_parse_tree(text))
    final_ans = list(final_ans)
    #print("REGEX",place_regex(text))
    #print("HOLO",place_holonym(text))
    #print("PARSE",place_parse_tree(text))
    return final_ans

#///////////////////////////////////////////////WORK TEMPLATE\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

def extract_person_pos_relations(doc):
    relations = []
    work_list = []
    for person in filter(lambda w: w.ent_type_ == 'PERSON', doc):
        work = ()
        pos_final = []
        org_final = []
        gpe_final = []
        if person.dep_ in ('nsubj'):
            position = [w for w in person.head.rights if w.dep_ == 'attr']
            if position:
                position = position[0]
                relations.append((person, position))
                pos_final.append(position)
                def extract_conjuncts(position):
                    for curr_pos in position.conjuncts:
                        relations.append((person, curr_pos))
                        pos_final.append(curr_pos)
                extract_conjuncts(position)
        elif person.dep_ == 'nsubj' and person.head.dep_ == 'root':
            position = [w for w in person.head.lefts if w.dep_ == 'nsubj']
            relations.append((person, position))
            pos_final.append(position)
        elif person.dep_ == 'ROOT':
            if len(list(doc.sents)) > 1:
                root = list(doc.sents)[1].root
                position = [w for w in root.rights if w.dep_ == 'attr']
                if position:
                    position = position[0]
                    relations.append((person, position))
                    pos_final.append(position)
                    def extract_conjuncts(position):
                        for curr_pos in position.conjuncts:
                            relations.append((person, curr_pos))
                            pos_final.append(curr_pos)
                    extract_conjuncts(position)
        for who in filter(lambda w: w.text.lower() == 'who'.lower(), doc):
            if who.dep_ == 'nsubj':
                who_prep = [w for w in who.head.rights if w.dep_ == 'prep' and w.text == 'as']
                if who_prep:
                    who_prep = who_prep[0]
                    position = [w for w in who_prep.rights if w.dep_ == 'pobj']
                    if position:
                        position = position[0]
                        relations.append((person, position))
                        pos_final.append(position)
        for pos in pos_final:
            pos_prep = [w for w in pos.rights if w.dep_ == 'prep']
            if pos_prep:
                pos_prep = pos_prep[0]
                org = [w for w in pos_prep.rights if w.ent_type_ == 'ORG']
                if org:
                    org = org[0]
                    org_final.append(org)
                if not org:
                    gpe = [w for w in pos_prep.rights if w.ent_type_ == 'GPE']
                    if gpe:
                        gpe = gpe[0]
                        gpe_final.append(gpe)
        for org in filter(lambda w: w.ent_type_ == 'GPE', doc):
            relations.append((person, org))
            org_final.append(org)
        if len(list(set(gpe_final))) == 1:
            gpe_final = gpe_final[0]
        work = (person, list(set(pos_final)), list(set(org_final)), gpe_final)
        work_list.append(work)
    return relations, work_list

def extract_person_pos_relations_table(doc):
    relations = []
    work_list = []
    for person in filter(lambda w: w.ent_type_ == 'PERSON', doc):
        work = ()
        pos_final = []
        org_final = []
        gpe_final = []
        if person.dep_ == 'ROOT':
            position = [w for w in person.rights if w.dep_ == 'attr' or w.dep_ == 'appos']
            if position:
                position = position[0]
                if position.ent_type_ == '':
                    relations.append((person, position))
                    pos_final.append(position)
                def extract_conjuncts(position):
                    for curr_pos in position.conjuncts:
                        if curr_pos.ent_type_ == '':
                            relations.append((person, curr_pos))
                            pos_final.append(curr_pos)
                extract_conjuncts(position)
        for org in filter(lambda w: w.ent_type_ == 'GPE', doc):
            relations.append((person, org))
            org_final.append(org)
        if len(list(set(gpe_final))) == 1:
            gpe_final = gpe_final[0]
        work = (person, list(set(pos_final)), list(set(org_final)), gpe_final)
        work_list.append(work)
    return relations, work_list




myFiles = glob.glob('*.txt')
for filename in myFiles:
    test_sentences=read_single_file_with_coref(filename)
    final_dict={}
    final_dict["document"]=filename
    final_dict["extraction"]=[]
    for sentence_text in test_sentences:
        try:
            sentence = nlp(sentence_text)
            sentence=merge_ents(sentence)
            #PART
            temp=place_template(sentence_text)
            if(temp!=[]):
                for j in temp:
                    temp_dict={}
                    temp_dict["template"]="PART"
                    temp_dict["sentences"]=[]
                    temp_dict["sentences"].append(sentence_text)
                    temp_dict["arguments"]={}
                    temp_dict["arguments"]["1"]=j[0]
                    temp_dict["arguments"]["2"]=j[1]
                    final_dict["extraction"].append(temp_dict)
            #BUY
            ans=extract_buy_templates(sentence)
            if(ans!=[]):
                for i in ans:
                    temp_dict={}
                    temp_dict["template"]="BUY"
                    temp_dict["sentences"]=[]
                    temp_dict["sentences"].append(sentence.text)
                    temp_dict["arguments"]={}
                    temp_dict["arguments"]["1"]=i["buyer"].text
                    temp_dict["arguments"]["2"]=i["item"].text
                    if(len(i["price"])==0):
                        temp_dict["arguments"]["3"]=i["price"]
                    else:
                        temp_dict["arguments"]["3"]=i["price"].text
                    if(len(i["quantity"])==0):
                        temp_dict["arguments"]["4"]=i["quantity"]
                    else:
                        temp_dict["arguments"]["4"]=i["quantity"].text
                    if(len(i["source"])==0):
                        temp_dict["arguments"]["5"]=i["source"]
                    else:
                        temp_dict["arguments"]["5"]=i["source"].text
                    final_dict["extraction"].append(temp_dict)
            #PERSON
            relations, work_list1 = extract_person_pos_relations(sentence)
            final_list=[]
            if(work_list1!=[]):
                for i in work_list1:
                    if i[1]!=[]:
                        final_list.append(i)
            if(final_list!=[]):
                for i in final_list:
                    temp_dict={}
                    temp_dict["template"]="WORK"
                    temp_dict["sentences"]=[]
                    temp_dict["sentences"].append(sentence.text)
                    temp_dict["arguments"]={}
                    temp_dict["arguments"]["1"]=i[0].text
                    stri=""
                    flag=True
                    for j in i[1]:
                        if(check_job(j.text)):
                            stri=stri+j.text+";"
                    if(stri!=""):
                        temp_dict["arguments"]["2"]=stri[:-1]
                    else:
                        flag=False
                    if(i[2]!=[]):
                        stri=""
                        for j in i[2]:
                            stri=stri+j.text+";"
                        temp_dict["arguments"]["3"]=stri[:-1]
                    else:
                        temp_dict["arguments"]["3"]=""
                    if(len(i[3])!=0):
                        stri=""
                        stri=stri+i[3].text+";"
                        temp_dict["arguments"]["4"]=stri[:-1]
                    else:
                        temp_dict["arguments"]["4"]=""

                    if(flag):
                        final_dict["extraction"].append(temp_dict)
        except: 
            continue
    test_sentences=read_single_file_table(filename)
    for sentence_text in test_sentences:
        try:
            sentence = nlp(sentence_text)
            sentence=merge_ents(sentence)
            #PERSON
            relations, work_list1 = extract_person_pos_relations_table(sentence)
            final_list=[]
            if(work_list1!=[]):
                for i in work_list1:
                    if i[1]!=[]:
                        final_list.append(i)
            if(final_list!=[]):
                for i in final_list:
                    temp_dict={}
                    temp_dict["template"]="WORK"
                    temp_dict["sentences"]=[]
                    temp_dict["sentences"].append(sentence.text)
                    temp_dict["arguments"]={}
                    temp_dict["arguments"]["1"]=i[0].text
                    stri=""
                    flag=True
                    for j in i[1]:
                        if(check_job(j.text)):
                            stri=stri+j.text+";"
                    if(stri!=""):
                        temp_dict["arguments"]["2"]=stri[:-1]
                    else:
                        flag=False
                    if(i[2]!=[]):
                        stri=""
                        for j in i[2]:
                            stri=stri+j.text+";"
                        temp_dict["arguments"]["3"]=stri[:-1]
                    else:
                        temp_dict["arguments"]["3"]=""
                    if(len(i[3])!=0):
                        stri=""
                        stri=stri+i[3].text+";"
                        temp_dict["arguments"]["4"]=stri[:-1]
                    else:
                        temp_dict["arguments"]["4"]=""
                    
                    if(flag):
                        final_dict["extraction"].append(temp_dict)
        except:
            continue

    json_filename="output "+filename[:len(filename)-4] + ".json"
    json_object = json.loads(json.dumps(final_dict))
    json_formatted_str = json.dumps(json_object, indent=4)

    file = open(json_filename, "w")
    n = file.write(json_formatted_str)
    file.close()
    print("JSON File for",filename," printed to current working directory")
print("Extraction Done")
