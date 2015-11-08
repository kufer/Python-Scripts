# -*- coding: utf-8 -*-
"""
Computings for Business Research, B9122 Fall 2014
Homework #7

Professor Yash Kanoria

Complete this web crawler:

This web crawler walks through the URLs in the source website. It only crawls 
child URLs if it contains at least one of the pre-defined key words. Each URL
has a score, which is defined as the number of occurence of the key words 
in the HTML body. 

As an output it prints out the sorted list of URLs based on the score.

@author: Kushal Fernandes
"""

import urllib.parse, urllib.request
import re
from bs4 import BeautifulSoup as bs
from collections import Counter

def html_words(soup):
    # A function that takes BeautifulSoup object as an input 
    # and converts the text body into a list of words.

    # a regular expression to remove non-alphanumeric characters
    regex = re.compile('[^a-zA-Z]') 

    text = soup.get_text()
    text = text.lower()
    text = regex.sub(' ', text)
    words = text.split()
    
    return words

def main():
    url = "http://www8.gsb.columbia.edu/"        
    maxNumUrl = 50      # the maximum number of URLs to be scanned
    keywords = ['finance', 'engineering', 'business', 'research']

    # initialize data arrays to be used
    """
        Hint: you may assign an arbitrary value to the parent score of
        "http://www8.gsb.columbia.edu/", which doesn't have a parent. It will
        be the first url opened in any case. 
        You will maintain a datastructure "urls" for unopened urls, another 
        one "opened" for opened urls, and possibly a third one "seen" that you 
        use to avoid repetition.
    """
    urls = {url : 0} 
    #dictionary to contain URL queue along with URL's parent score
    opened = {url : 0}
    #dictionary to contain opened URLs and their score
    seen = []
    #list of all seen URLs so that we do not have duplicates in URLs queue
    
    #repeat as long as URL queue is not empty or maxNumUrl is not exceeded
    while len(urls) > 0 and len(opened) < maxNumUrl:
        
        #set the current URL as the url with the highest parent score
        curr_url = max(urls, key = urls.get)
        #delete the current URL from the urls queue
        del urls[curr_url] 
        
        #open the current URL and get its htmltext
        try:        
            htmltext = urllib.request.urlopen(curr_url).read()
        except Exception as ex:
            print(ex)
            continue
        
        # convert the URL's htmltext into a list of words
        soup = bs(htmltext)
        words = html_words(soup)
        
        # get score of the current URL based on keyword occurences in its HTML
        score = sum([Counter(words)[keyword] for keyword in keywords]) 
        
        #if the current URL has a score greater than 0, we find its child URLs
        #and add them to queue of URLs along with the score of the current URL
        #which is their parent URL. This is done only if the child URL has not
        #been seen before so as to avoid repitition in the URLs queue. If
        #the child URL is unique it is also added to the seen list.
        #Finally, the current URL which is the parent of all these child URLs
        #is added to the opened queue along with its newly calculated score
        if score > 0:
            for tag in soup.find_all('a', href = True):
                childUrl = tag['href']
                childUrl = urllib.parse.urljoin(url, childUrl)
                if url in childUrl and childUrl not in seen:
                    urls[childUrl] = score #add to URL queue with parent score
                    seen.append(childUrl) #add to list of seen URLs
        
        opened[curr_url] = score #add current URL with score to opened list
        
        
   # print top 10 list of opened URLs with the highest scores
    print(*sorted(opened.items(), key = lambda item: item[1], reverse = True)[0:10], sep = '\n')

if __name__ == '__main__':
    main()