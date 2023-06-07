from flask import Flask, render_template, request
import pandas as pd
import statistics
import numpy as np
import requests
from bs4 import BeautifulSoup
import csv
import re
import time
import pickle, joblib
import nltk
from tpot import TPOTClassifier
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from newspaper import Article
from sklearn.preprocessing import MinMaxScaler

def remove_special_characters(text):
    # Remove special characters and punctuation using regular expressions
    cleaned_text = re.sub(r'[^\d\s.-]', '', text)
    return cleaned_text

def convert_relative_time(relative_time):
    pattern = r'(\d+)\s+(\w+)\s+ago'
    match = re.search(pattern, relative_time)
    if match:
        count = int(match.group(1))
        unit = match.group(2)
        if unit == 'days':
            return timedelta(days=count)
        elif unit == 'hours':
            return timedelta(hours=count)
        elif unit == 'minutes':
            return timedelta(minutes=count)
    return timedelta()

def get_company_news(url):
    # Retrieve and parse the company news page
    # ...
    company_news_page = requests.get(url)
    if company_news_page.status_code == 200:
        print("Success : Requested ticker tape news page is loaded successfully ")
        company_news_content = company_news_page.content
        comp_news_soup = BeautifulSoup(company_news_content,"html.parser")
        leftsidebar = comp_news_soup.find('div', attrs ={'class':'desktop-side-panel desktop--only'})
        CompanyName = leftsidebar.find('span', attrs ={'class':'jsx-3488654145 ticker text-teritiary font-medium'}).get_text()
        FullCompanyName = leftsidebar.find('div', attrs ={'class':'jsx-3488654145 full-width d-flex justify-space-between sidebar-security-name'}).get_text()
        news_section = comp_news_soup.find('div', attrs ={'class':'jsx-4278471340'})          
        news_cards = news_section.select('.latest-news-holder a')
        sentiment_score = []
        for card in news_cards:
            NewsURL = card['href']
            NewsTitle = card.select_one('.news-title').text.strip()
            jsx_value = card['class'][1]  # Assuming the `jsx-3953764037` class is always the second class
            NewsDate = card.select_one('.news-info span').text.strip()
            NewsDate = datetime.now() - convert_relative_time(NewsDate)
            print("Full Company Name:",FullCompanyName)
            print("Company Name:",CompanyName)
            print("News URL:", NewsURL)
            print("News Title:", NewsTitle)
            print("News Date:", NewsDate)
            news_article = Article(NewsURL) #providing the link
            try:
              news_article.download()
              news_article.parse()
              news_article.nlp()
            except:
               pass 
            NewsArticle=news_article.text
            NewsSummary=news_article.summary
            print("News Article:",NewsArticle)
            print("News Summary:", NewsSummary) 
            
            sia = SentimentIntensityAnalyzer()
            # Apply sentiment analysis to each news headline
            sentiment_score_compound = sia.polarity_scores(NewsArticle)['compound']
            print("Sentiment Score of News:", sentiment_score_compound)
            if(NewsDate > pd.Timestamp.now() - pd.Timedelta(days=2)):
                sentiment_score.append(sentiment_score_compound)
        
    # Return the company news content
    return sentiment_score

def calculate_average_sentiment(sentiment_scores):
    if not sentiment_scores:
        # Return a default sentiment value or handle the empty case as needed
        return 0
    # Calculate the average sentiment score
    avg_sentiment_score = statistics.mean(sentiment_scores)
    print("Average Sentiment Score:", avg_sentiment_score)

    # Return the average sentiment score whose value are from -1(negative) to +1(positive)
    return avg_sentiment_score

def get_header_agent():
    # Prepare the header agent
    # ...
    header_agent = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Connection': 'keep-alive'}
    # Return the header agent
    return header_agent
# Function to assign sentiment labels based on keywords
def assign_sentiment_label(text):
    lowercase_text = text.lower()
    if 'high' in lowercase_text:
        return 2
    elif 'avg' in lowercase_text:
        return 1
    else:
        return 0

def redflag_valuation_sentiment_label(text):
    lowercase_text = text.lower()
    if 'high' in lowercase_text:
        return 0
    elif 'avg' in lowercase_text:
        return 1
    else:
        return 2

def entrypoint_sentiment_label(text):
    lowercase_text = text.lower()
    if 'bad' in lowercase_text:
        return 0
    elif 'good' in lowercase_text:
        return 2
    else:
        return 1

def get_company_data(url_val_overview,url_val_finance, header_agent, avg_sentiment):
    # Retrieve and parse the company overview page
    # ...
    company_ovr_page = requests.get(url_val_overview,headers=header_agent,stream=True)
    print(url_val_overview)
    if company_ovr_page.status_code == 200:
        print("Success : Requested ticker tape overview page is loaded successfully ")
        company_ovr_content = company_ovr_page.content
        comp_ovr_soup = BeautifulSoup(company_ovr_content,"html.parser")
        part_1 = comp_ovr_soup.find('div', attrs ={'class':'jsx-3488654145 sidebar desktop--only stock-security-sidebar'})
        CompanyName = part_1.find('span', attrs ={'class':'jsx-3488654145 ticker text-teritiary font-medium'}).get_text()
        FullCompanyName = part_1.find('div', attrs ={'class':'jsx-3488654145 full-width d-flex justify-space-between sidebar-security-name'}).get_text()
        print(CompanyName)
        print(FullCompanyName)
        key_metrics = comp_ovr_soup.find('div', attrs ={'class':'jsx-3519906982 stat-table-wrapper'})
        count=0
        for row in key_metrics.thead.find_all('tr'):    
            columns = row.find_all('th')
            for column in columns:
                if(column.text == "No LabelNo Label"):
                    i=count
                count=count+1
        for row in key_metrics.tbody.find_all('tr'):    
            columns = row.find_all('td')
            PERatio=columns[i].text
            print(PERatio)
        try:
            scorecard = comp_ovr_soup.find('div', attrs ={'class':'jsx-1630544676 scorecard-container relative'})
            string = scorecard.get_text()
            soup = True
        except:
            soup = False
    company_fin_page = requests.get(url_val_finance,headers=header_agent,stream=True)
    print(url_val_finance)
    if company_fin_page.status_code == 200 :
        print("Success : Requested ticker tape financial page is loaded successfully ")
        company_fin_content = company_fin_page.content
        comp_fin_soup = BeautifulSoup(company_fin_content,"html.parser")
        if soup == False:
            try:
                scorecard = comp_fin_soup.find('div', attrs ={'class':'jsx-1630544676 scorecard-container relative'})
                string = scorecard.get_text()
                soup = True
            except:
                soup = False
        if soup == True:
            string = string.replace("Scorecard","" , 1)
            string = string.replace("Performance","\nPerformance ")
            string = string.replace("Valuation","\nValuation ")
            string = string.replace("Growth","\nGrowth ")
            string = string.replace("Profitability","\nProfitability ")
            string = string.replace("Entry point","\nEntry point ")
            string = string.replace("Red flags","\nRed flags ")
            string = string.replace("Avg"," Avg ")
            string = string.replace("High"," High ")
            string = string.replace("Low"," Low ")
            string = string.replace("Good"," Good ")
            sc_array = re.split('\n',string)
            for i in sc_array:
                split_sc_array = i.split(" ",1)
                if(split_sc_array[0] == "Performance" ):
                    sc_Performance=split_sc_array[1]
                elif(split_sc_array[0] == "Valuation" ):
                    sc_Valuation=split_sc_array[1]
                elif(split_sc_array[0] == "Growth" ):
                    sc_Growth=split_sc_array[1]
                elif(split_sc_array[0] == "Profitability" ):
                    sc_Profitability=split_sc_array[1]
                elif(split_sc_array[0] == "Entry" ):
                    sc_Entrypoint=split_sc_array[1].split(" ",1)
                    sc_Entrypoint = sc_Entrypoint[1]
                elif(split_sc_array[0] == "Red" ):
                    sc_Redflags=split_sc_array[1].split(" ",1)
                    sc_Redflags=sc_Redflags[1]
            print(sc_Performance)
            print(sc_Valuation)
            print(sc_Growth)
            print(sc_Profitability)
            print(sc_Entrypoint)
            print(sc_Redflags)  
            sc_Entrypoint_sentiment = entrypoint_sentiment_label(sc_Entrypoint)
            sc_Growth_sentiment = assign_sentiment_label(sc_Growth)
            sc_Profitability_sentiment = assign_sentiment_label(sc_Profitability)
            sc_Redflags_sentiment = redflag_valuation_sentiment_label(sc_Redflags)
            sc_Performance_sentiment = assign_sentiment_label(sc_Performance)
            sc_Valuation_sentiment = redflag_valuation_sentiment_label(sc_Valuation)
        try:
            IncomeStatement= comp_fin_soup.find('div',attrs ={'class':'jsx-2537935686 commentary-items'}).text
        except:
            IncomeStatement= "Income statement not found"
        print(IncomeStatement)
        table = comp_fin_soup.find('table',class_='jsx-2597786574 jsx-1728146729')
        count=0
        i2019=i2020=i2021=i2022=-1
        try:
            for row in table.thead.find_all('tr'):    
                columns = row.find_all('th')
                for column in columns:
                    if(column.text == "FY 2019"):
                        i2019=count
                    elif(column.text == "FY 2020"):
                        i2020=count
                    elif(column.text == "FY 2021"):
                        i2021=count
                    elif(column.text == "FY 2022"):
                        i2022=count
                    count=count+1
            for row in table.tbody.find_all('tr'):    
                columns = row.find_all('td')
                if(columns[0].text == "Total Revenue"):
                    if(i2019 != -1):
                        TotalRevenue2019=columns[i2019].text
                    else:
                        TotalRevenue2019=0 
                    if(i2020 != -1):
                        TotalRevenue2020=columns[i2020].text
                    else:
                        TotalRevenue2020=0 
                    if(i2021 != -1):
                        TotalRevenue2021=columns[i2021].text
                    else:
                        TotalRevenue2021=0 
                    if(i2022 != -1):
                        TotalRevenue2022=columns[i2022].text
                    else:
                        TotalRevenue2022=0 
                elif(columns[0].text == "EBITDA"):
                    if(i2019 != -1):
                        EBITDA2019=columns[i2019].text
                    else:
                        EBITDA2019=0
                    if(i2020 != -1):
                        EBITDA2020=columns[i2020].text
                    else:
                        EBITDA2020=0
                    if(i2021 != -1):
                        EBITDA2021=columns[i2021].text
                    else:
                        EBITDA2021=0
                    if(i2022 != -1):
                        EBITDA2022=columns[i2022].text
                    else:
                        EBITDA2022=0
                elif(columns[0].text == "Net Income"):
                    if(i2019 != -1):
                        NetIncome2019=columns[i2019].text
                    else:
                        NetIncome2019=0
                    if(i2020 != -1):
                        NetIncome2020=columns[i2020].text
                    else:
                        NetIncome2020=0
                    if(i2021 != -1):
                        NetIncome2021=columns[i2021].text
                    else:
                        NetIncome2021=0
                    if(i2022 != -1):
                        NetIncome2022=columns[i2022].text
                    else:
                        NetIncome2022=0
                elif(columns[0].text == "PBT"):
                    if(i2019 != -1):
                        PBT2019=columns[i2019].text
                    else:
                        PBT2019=0
                    if(i2020 != -1):
                        PBT2020=columns[i2020].text
                    else:
                        PBT2020=0
                    if(i2021 != -1):
                        PBT2021=columns[i2021].text
                    else:
                        PBT2021=0
                    if(i2022 != -1):
                        PBT2022=columns[i2022].text
                    else:
                        PBT2022=0
        except:
            TotalRevenue2019=0
            TotalRevenue2020=0
            TotalRevenue2021=0
            TotalRevenue2022=0
            EBITDA2019=0
            EBITDA2020=0
            EBITDA2021=0
            EBITDA2022=0
            NetIncome2019=0
            NeIncome2020=0
            NetIncome2021=0
            NetIncome2022=0
            PBT2019=0
            PBT2020=0
            PBT2021=0
            PBT2022=0
        print(TotalRevenue2019)
        print(TotalRevenue2020)
        print(TotalRevenue2021)
        print(TotalRevenue2022)
        print(EBITDA2019)
        print(EBITDA2020)
        print(EBITDA2021)
        print(EBITDA2022)
        print(NetIncome2019)
        print(NetIncome2020)
        print(NetIncome2021)
        print(NetIncome2022)
        print(PBT2019)
        print(PBT2020)
        print(PBT2021)
        print(PBT2022)
        
        

    flagdata = {
        '2019 Total Revenue' : TotalRevenue2019,
        '2019 EBITDA' : EBITDA2019,
        '2019 Net Income' : NetIncome2019,
        '2019 PBT' : PBT2019,
        '2020 Total Revenue' : TotalRevenue2020,
        '2020 EBITDA' : EBITDA2020,
        '2020 Net Income' : NetIncome2020,
        '2020 PBT' : PBT2020,
        '2021 Total Revenue' : TotalRevenue2021,
        '2021 EBITDA' : EBITDA2021,
        '2021 Net Income' : NetIncome2021,
        '2021 PBT' : PBT2021,
        '2022 Total Revenue' : TotalRevenue2022,
        '2022 EBITDA' : EBITDA2022,
        '2022 Net Income' : NetIncome2022,
        '2022 PBT' : PBT2022,
        'PE Ratio' : PERatio,
        'Entry_point_Sentiment' : sc_Entrypoint_sentiment,
        'Growth_Sentiment' : sc_Growth_sentiment,
        'Profitability_Sentiment' : sc_Profitability_sentiment,
        'RedFlag_Sentiment' : sc_Redflags_sentiment,
        'Performance_Sentiment' : sc_Performance_sentiment,
        'Valuation_Sentiment' : sc_Valuation_sentiment,
        'News Sentiment Score': avg_sentiment,
        'Final_Sentiment':'1'
    }
    # Convert the dictionary to a DataFrame
    flagdf = pd.DataFrame(flagdata, index=[0])
        
       
    # Return the company Full data in Dataframe
    return flagdf

def process_data(flagdf):
    # Process the data and create the flagdf DataFrame
    print(flagdf.T)
    #flagdf = flagdf.fillna(0)

    # ...
    print("-------remove special---------")
    flagdf['2019 Total Revenue']= flagdf['2019 Total Revenue'].apply(remove_special_characters)
    flagdf['2019 EBITDA']= flagdf['2019 EBITDA'].apply(remove_special_characters)
    flagdf['2019 Net Income']= flagdf['2019 Net Income'].apply(remove_special_characters)
    flagdf['2019 PBT']= flagdf['2019 PBT'].apply(remove_special_characters)
    flagdf['2020 Total Revenue']= flagdf['2020 Total Revenue'].apply(remove_special_characters)
    flagdf['2020 EBITDA']= flagdf['2020 EBITDA'].apply(remove_special_characters)
    flagdf['2020 Net Income']= flagdf['2020 Net Income'].apply(remove_special_characters)
    flagdf['2020 PBT']= flagdf['2020 PBT'].apply(remove_special_characters)
    flagdf['2021 Total Revenue']= flagdf['2021 Total Revenue'].apply(remove_special_characters)
    flagdf['2021 EBITDA']= flagdf['2021 EBITDA'].apply(remove_special_characters)
    flagdf['2021 Net Income']= flagdf['2021 Net Income'].apply(remove_special_characters)
    flagdf['2021 PBT']= flagdf['2021 PBT'].apply(remove_special_characters)
    flagdf['2022 Total Revenue']= flagdf['2022 Total Revenue'].apply(remove_special_characters)
    flagdf['2022 EBITDA']= flagdf['2022 EBITDA'].apply(remove_special_characters)
    flagdf['2022 Net Income']= flagdf['2022 Net Income'].apply(remove_special_characters)
    flagdf['2022 PBT']= flagdf['2022 PBT'].apply(remove_special_characters)
    flagdf['PE Ratio']= flagdf['PE Ratio'].apply(remove_special_characters)
    print(flagdf.T)
    print("-------After remove special---------")
    X_values = pd.read_csv('X_values.csv')
    X_values.columns = ['News Sentiment Score','2019 Total Revenue','2019 EBITDA','2019 Net Income','2019 PBT',
                        '2020 Total Revenue','2020 EBITDA','2020 Net Income','2020 PBT',
                        '2021 Total Revenue','2021 EBITDA','2021 Net Income','2021 PBT',
                        '2022 Total Revenue','2022 EBITDA','2022 Net Income','2022 PBT','PE Ratio']
    print(X_values)
    scaler = MinMaxScaler(feature_range=(0, 2))
    
    # scaler.fit(X_values)
    scaler.fit(X_values)
    print("datamax",scaler.data_max_)
    print("datamin",scaler.data_min_)
    columns_to_scale = ['News Sentiment Score','2019 Total Revenue','2019 EBITDA','2019 Net Income','2019 PBT',
                        '2020 Total Revenue','2020 EBITDA','2020 Net Income','2020 PBT',
                        '2021 Total Revenue','2021 EBITDA','2021 Net Income','2021 PBT',
                        '2022 Total Revenue','2022 EBITDA','2022 Net Income','2022 PBT','PE Ratio']

    scaled_values = scaler.transform(flagdf[columns_to_scale])
    scaled_df = pd.DataFrame(scaled_values, columns=columns_to_scale)
    flagdf[columns_to_scale] = scaled_df[columns_to_scale]

    print("--------After minmax scaler------")
    print(flagdf.T)
    model = joblib.load(open('Sentiment_Analysis_Flag_Data.pkl', 'rb'))
    predictions = pd.DataFrame(model.predict(flagdf), columns = ['prediction'])
    
    final = pd.concat([predictions, flagdf], axis = 1)
    print("--------After Prediction scaler------")
    print(final)
    # Return the final DataFrame
    return final

app = Flask(__name__, static_folder='static')

# Load the DataFrame once outside the route functions
df = pd.read_csv('companyListURL.tsv', delimiter='\t', encoding='utf-8')

@app.route('/')
def index():
    column_values = df['Company Name'].tolist()
    return render_template('frontend.html', response="Select a Company",column_names=column_values)
@app.route('/', methods=['POST'])
def submit():
    column_values = df['Company Name'].tolist()
    company_name = request.form['company_name']
    if company_name is not None:
        url_val = df.loc[df['Company Name'] == company_name, 'URL'].values[0]
        print(url_val)
        url_val_news=url_val + "/news?checklist=basic&type=news"
        print(url_val_news)
        
        company_news_content = get_company_news(url_val_news)
        avg_sentiment = calculate_average_sentiment(company_news_content)
        header_agent = get_header_agent()

        url_val_finance=url_val + "/financials?checklist=basic&statement=income&view=normal&period=annual"
        url_val_overview=url_val + "?checklist=basic&chartScope=1d"
        flagdf = get_company_data(url_val_overview, url_val_finance, header_agent, avg_sentiment)
        final = process_data(flagdf)
            #in diagnosis 0 is fairly Negative , 1 as fairly positive, as negative and ,3 as positive
            # Sentiment_Category values are 'Positive'as 2, 'Negative'as 0, or 'Neutral'as 1 based on the sentiment score
        if final['prediction'].iloc[0] == 0:
            response = "The sentiment is Negative."   
        elif final['prediction'].iloc[0] == 1:
            response = "The sentiment is Neutral."  
        else:
            response = "The sentiment is Positive."

    else:
        response = "Select a Company"

    return render_template('frontend.html', response=response,column_names=column_values)

if __name__ == '__main__':
    app.run()