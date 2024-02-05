import random

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from tqdm import tqdm
import time

chromeOptions = Options()
chromeOptions.add_argument('--headless')
driver = webdriver.Chrome(options=chromeOptions)
driver.implicitly_wait(5)

def pause():
    time.sleep(random.randrange(1, 4))

with open('Data/Data_from_Scraping.csv', 'r') as file:
    lines = [x.strip() for x in file.readlines()]
    words = [x.split(';')[0] for x in lines]
    already_scraped = [x.split(';')[0].strip() for x in lines if x.split(';')[1]]



driver.get('https://de.pons.com/übersetzung')
time.sleep(3)
driver.maximize_window()
time.sleep(1)
driver.find_element(By.XPATH, "//button[text()='Mehr Optionen']").click()
pause()
driver.find_element(By.XPATH, "//button[text()='Einstellungen speichern']").click()
pause()
search_bar = driver.find_element(By.XPATH, "//input[@class='input-large pons-search-input']")
search_bar.send_keys('Test')
driver.find_element(By.XPATH, "//a[@class='btn btn-primary btn-large submit']").click()
pause()
for word in tqdm(words):
    if word not in already_scraped:
        search_bar = driver.find_element(By.XPATH, "//input[@class='pons-search-input']")
        search_bar.send_keys(word)
        time.sleep(1)
        driver.find_element(By.XPATH, "//a[@class='btn btn-primary submit']").click()
        pause()
        if driver.find_element(By.CSS_SELECTOR, 'div.lang_dir > span.flag').get_attribute('class') == 'flag flag_de':
            with open('Data/Data_from_Scraping.csv', 'a') as file:
                try:
                    file.write(f"{word};{driver.find_element(By.XPATH, '//h2').text}\n")
                except NoSuchElementException:
                    file.write(f"NoSuchElementException{word}")
        else:
            with open('Data/Data_from_Scraping.csv', 'a') as file:
                file.write(f'{word};ACHTUNG!!! lang_dir = {driver.find_element(By.CSS_SELECTOR, 'div.lang_dir > span.flag').get_attribute('class')}\n')

#text = driver.find_element(By.XPATH, "//h2").text
#print(text)
driver.quit()