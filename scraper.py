"""
Use selenium to repeated click the "show more" button on the all professors page.

Then download the HTML and scrape as many profID as possible.
"""

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time


driver = webdriver.Chrome('/Users/jackyu/Desktop/486/chromedriver_mac64/chromedriver')
driver.get('https://www.ratemyprofessors.com/search/teachers?query=*&sid=1258')
xpath = '//*[@id="root"]/div/div/div[4]/div[1]/div[1]/div[4]/button'
profxpath = '//*[@id="root"]/div/div/div[4]/div[1]/div[1]/div[3]/a[16]/div/div[2]/div[1]'

count = 1
while True:
    try:
        WebDriverWait(driver, 180).until(EC.visibility_of_element_located((By.XPATH, xpath))).click()
        # loadmore = driver.find_element("xpath", xpath)
        time.sleep(1)
        print("button clicked " + str(count))
        count += 1
    except Exception as e:
        print(e)
        break
f = open("output.txt", "w")
h = driver.page_source
f.write(h)
