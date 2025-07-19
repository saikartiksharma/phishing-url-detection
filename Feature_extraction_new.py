
from urllib.parse import urlparse,urlencode,parse_qs
from bs4 import BeautifulSoup
from googlesearch import search
import requests
import socket
import re
import pandas as pd
import whois
from datetime import date, datetime
import time
from dateutil.parser import parse as date_parse






def having_IP_Address(url):
    domain = urlparse(url).netloc
    try:
        ip = socket.gethostbyname(domain)
        return 1
    except:
        return -1


def URL_Length(url):
    if len(url) < 54:
        return 1
    elif len(url) >= 54 and len(url) <= 75:
        return 0
    else:
        return -1




def Shortining_Service(url):
    match=re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                        'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                        'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                        'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                        'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                        'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                        'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net',url)
    if match:
        return -1
    else:
        return 1





def having_At_Symbol(url):
    if re.findall("@", url):
        return -1
    else:
        return 1





def double_slash_redirecting(url):
    list=[x.start(0) for x in re.finditer('//', url)]
    if list[len(list)-1]>6:
        return -1
    else:
        return 1




def Prefix_Suffix(url):
    if re.findall(r"https?://[^\-]+-[^\-]+/", url):
        return -1
    else:
        return 1




def having_Sub_Domain(url):
    if len(re.findall("\.", url)) == 1:
        return 1
    elif len(re.findall("\.", url)) == 2:
        return 0
    else:
        return -1


def SSLfinal_State(url,response):
    try:
        if response.text:
            return 1
    except:
        return -1




def Domain_registeration_length(url):
    domain_info = whois.whois(url)
    expiration_date = domain_info.expiration_date
    registration_length = 0
    try:
        expiration_date = min(expiration_date)
        today = time.strftime('%Y-%m-%d')
        today = datetime.strptime(today, '%Y-%m-%d')
        registration_length = abs((expiration_date - today).days)
        if registration_length <= 365:
            return -1
        else:
            return 1
    except:
        return 1




def Favicon(url,soup,domain):
    if soup == -999:
        return -1
    else:
        try:
            for head in soup.find_all('head'):
                for head.link in soup.find_all('link', href=True):
                    dots = [x.start(0) for x in re.finditer('\.', head.link['href'])]
                    if url in head.link['href'] or len(dots) == 1 or domain in head.link['href']:
                        return 1
                    else:
                        return -1
                      
        except StopIteration:
            pass




def port(url,domain):
    try:
        port = domain.split(":")[1]
        if port:
            return -1
        else:
            return 1
    except:
        return 1




def HTTPS_token(url):
    if re.findall(r"^https://", url):
        return 1
    else:
        return -1


def Request_URL(url,soup,domain):
    i = 0
    success = 0
    if soup == -999:
        return -1
    else:
        for img in soup.find_all('img', src= True):
            dots= [x.start(0) for x in re.finditer('\.', img['src'])]
            if url in img['src'] or domain in img['src'] or len(dots)==1:
                success = success + 1
            i=i+1

        for audio in soup.find_all('audio', src= True):
            dots = [x.start(0) for x in re.finditer('\.', audio['src'])]
            if url in audio['src'] or domain in audio['src'] or len(dots)==1:
                success = success + 1
            i=i+1

        for embed in soup.find_all('embed', src= True):
            dots=[x.start(0) for x in re.finditer('\.',embed['src'])]
            if url in embed['src'] or domain in embed['src'] or len(dots)==1:
                success = success + 1
            i=i+1

        for iframe in soup.find_all('iframe', src= True):
            dots=[x.start(0) for x in re.finditer('\.',iframe['src'])]
            if url in iframe['src'] or domain in iframe['src'] or len(dots)==1:
                success = success + 1
            i=i+1

        try:
            percentage = success/float(i) * 100
            if percentage < 22.0 :
                return 1
            elif((percentage >= 22.0) and (percentage < 61.0)) :
                return 0
            else :
                return -1
        except:
                return 1




def URL_of_Anchor(url,soup,domain):
    percentage = 0
    i = 0
    unsafe=0
    if soup == -999:
        return -1
    else:
        for a in soup.find_all('a', href=True):
          # 2nd condition was 'JavaScript ::void(0)' but we put JavaScript because the space between javascript and :: might not be
                  # there in the actual a['href']
            if "#" in a['href'] or "javascript" in a['href'].lower() or "mailto" in a['href'].lower() or not (url in a['href'] or domain in a['href']):
                unsafe = unsafe + 1
            i = i + 1


        try:
            percentage = unsafe / float(i) * 100
        except:
            return 1

        if percentage < 31.0:
            return 1
        elif ((percentage >= 31.0) and (percentage < 67.0)):
            return 0
        else:
            return -1



def Links_in_tags(url,soup,domain):
    i=0
    success =0
    if soup == -999:
        return -1
    else:
        for link in soup.find_all('link', href= True):
            dots=[x.start(0) for x in re.finditer('\.',link['href'])]
            if url in link['href'] or domain in link['href'] or len(dots)==1:
                success = success + 1
            i=i+1

        for script in soup.find_all('script', src= True):
            dots=[x.start(0) for x in re.finditer('\.',script['src'])]
            if url in script['src'] or domain in script['src'] or len(dots)==1 :
                success = success + 1
            i=i+1
        try:
            percentage = success / float(i) * 100
        except:
            return 1

        if percentage < 17.0 :
            return 1
        elif((percentage >= 17.0) and (percentage < 81.0)) :
            return 0
        else :
            return -1




def Submitting_to_email(url,response):
    if response == "":
        return -1
    else:
        if re.findall(r"[mail\(\)|mailto:?]", response.text):
            return 1
        else:
            return-1


def Abnormal_URL(url,response):
    if response == "":
        return -1
    else:
        if response.text == "":
            return 1
        else:
            return -1



def Redirect(url,response):
    if response == "":
        return -1
    
    else:
        if len(response.history) <= 1:
            return -1
        elif len(response.history) <= 4:
            return 0
        else:
            return 1




def on_mouseover(url,response):
    if response == "" :
        return -1
    else:
        if re.findall("<script>.+onmouseover.+</script>", response.text):
            return 1
        else:
            return -1


def RightClick(url,response):
    if response == "":
        return -1
    else:
        if re.findall(r"event.button ?== ?2", response.text):
            return 1
        else:
            return -1




def popUpWidnow(url,response):
    if response == "":
        return -1
    else:
        if re.findall(r"alert\(", response.text):
            return 1
        else:
            return -1



def Iframe(url,response):
    if response == "":
        return -1
    else:
        if re.findall(r"[<iframe>|<frameBorder>]", response.text):
            return 1
        else:
            return -1


def diff_month(d1, d2):
        return (d1.year - d2.year) * 12 + d1.month - d2.month

def age_of_domain(url,response):
    if response == "":
        return  -1
    else:
        try:
            registration_date = re.findall(r'Registration Date:</div><div class="df-value">([^<]+)</div>', response.text)[0]
            if diff_month(date.today(), date_parse(registration_date)) >= 6:
                return -1
            else:
                return 1
        except:
            return 1



def DNSRecord(url,domain):
    dns = 1
    try:
        domain_info = whois.whois(domain)
        expiration_date = domain_info.expiration_date
        registration_length = 0
        expiration_date = min(expiration_date)
        today = time.strftime('%Y-%m-%d')
        today = datetime.strptime(today, '%Y-%m-%d')
        registration_length = abs((expiration_date - today).days)
 # Initialize registration_length here
    except:
        dns=-1
    if dns == -1:
        return -1
    else:
        if registration_length / 365 <= 1:
            return -1
        else:
            return 1
        



def Page_Rank(url,google_rank):
        if google_rank > 0 and google_rank <= 10:
            return -1
        else:
            return 1



def Google_Index(url):
    site = search(url, 5)
    if site:
        return 1
    else:
        return -1




def Links_pointing_to_page(url,response):
    if response == "":
        return -1
    else:
        number_of_links = len(re.findall(r"<a href=", response.text))
        if number_of_links == 0:
            return 1
        elif number_of_links <= 2:
            return 0
        else:
            return -1




def Statistical_report(url,domain):

    try:
        ip_address=socket.gethostbyname(domain)
        ip_match=re.search('146\.112\.61\.108|213\.174\.157\.151|121\.50\.168\.88|192\.185\.217\.116|78\.46\.211\.158|181\.174\.165\.13|46\.242\.145\.103|121\.50\.168\.40|83\.125\.22\.219|46\.242\.145\.98|'
                              '107\.151\.148\.44|107\.151\.148\.107|64\.70\.19\.203|199\.184\.144\.27|107\.151\.148\.108|107\.151\.148\.109|119\.28\.52\.61|54\.83\.43\.69|52\.69\.166\.231|216\.58\.192\.225|'
                              '118\.184\.25\.86|67\.208\.74\.71|23\.253\.126\.58|104\.239\.157\.210|175\.126\.123\.219|141\.8\.224\.221|10\.10\.10\.10|43\.229\.108\.32|103\.232\.215\.140|69\.172\.201\.153|'
                              '216\.218\.185\.162|54\.225\.104\.146|103\.243\.24\.98|199\.59\.243\.120|31\.170\.160\.61|213\.19\.128\.77|62\.113\.226\.131|208\.100\.26\.234|195\.16\.127\.102|195\.16\.127\.157|'
                              '34\.196\.13\.28|103\.224\.212\.222|172\.217\.4\.225|54\.72\.9\.51|192\.64\.147\.141|198\.200\.56\.183|23\.253\.164\.103|52\.48\.191\.26|52\.214\.197\.72|87\.98\.255\.18|209\.99\.17\.27|'
                              '216\.38\.62\.18|104\.130\.124\.96|47\.89\.58\.141|78\.46\.211\.158|54\.86\.225\.156|54\.82\.156\.19|37\.157\.192\.102|204\.11\.56\.48|110\.34\.231\.42',ip_address)
        if ip_match:
                return -1
        else:
                return 1
    except:
            return -1
        


def get_data_set(url):

    try:
        response = requests.get(url)
    except:
        response =""
    
    try:
        soup = BeautifulSoup(response.text, 'html.parser')
    except:
        soup = -999
    
    domain = urlparse(url).netloc
    if re.match(r"^www.",domain):
        domain = domain.replace("www.","")


    dataset = []

    


    
        # Extracts google rank of the website
    try:
        rank_checker_response = requests.post("https://www.checkpagerank.net/index.php", {"name": domain})
        google_rank = int(re.findall(r"Google PageRank: <span style=\"color:#000099;\">([0-9]+)", rank_checker_response.text)[0], timeout=5)
    except:
        google_rank = -1
    


    dataset.append(having_IP_Address(url))                       #1
    dataset.append(URL_Length(url))                              #2
    dataset.append(Shortining_Service(url))                      #3
    dataset.append(having_At_Symbol(url))                        #4
    dataset.append(double_slash_redirecting(url))                #5
    dataset.append(Prefix_Suffix(url))                           #6
    dataset.append(having_Sub_Domain(url))                       #7
    dataset.append(SSLfinal_State(url,response))                 #8
    dataset.append(Domain_registeration_length(url))             #9
    dataset.append(Favicon(url,soup,domain))                     #10
    dataset.append(port(url,domain))                             #11
    dataset.append(HTTPS_token(url))                             #12
    dataset.append(Request_URL(url,soup,domain))                 #13
    dataset.append(URL_of_Anchor(url,soup,domain))               #14
    dataset.append(Links_in_tags(url,soup,domain))               #15
    dataset.append(Submitting_to_email(url,response))            #16
    dataset.append(Abnormal_URL(url,response))                   #17
    dataset.append(Redirect(url,response))                       #18
    dataset.append(on_mouseover(url,response))                   #19
    dataset.append(RightClick(url,response))                     #20
    dataset.append(popUpWidnow(url,response))                    #21
    dataset.append(Iframe(url,response))                         #22
    dataset.append(age_of_domain(url,response))                  #23
    dataset.append(DNSRecord(url,domain))                        #24 
    dataset.append(Page_Rank(url,google_rank))                   #25
    dataset.append(Google_Index(url))                            #26
    dataset.append(Links_pointing_to_page(url,response))         #27
    dataset.append(Statistical_report(url,domain))               #28
    
    data = pd.DataFrame(dataset).T

    data.columns = ['having_IP_Address','URL_Length','Shortining_Service','having_At_Symbol','double_slash_redirecting','Prefix_Suffix','having_Sub_Domain',
                    'SSLfinal_State','Domain_registeration_length','Favicon','port','HTTPS_token','Request_URL','URL_of_Anchor','Links_in_tags','Submitting_to_email',
                    'Abnormal_URL','Redirect','on_mouseover','RightClick','popUpWidnow','Iframe','age_of_domain','DNSRecord','Page_Rank','Google_Index',
                    'Links_pointing_to_page','Statistical_report']
    
    dataset = []
    return data
