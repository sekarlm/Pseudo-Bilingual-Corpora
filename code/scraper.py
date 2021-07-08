from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

PATH = 'C:\Program Files (x86)\chromedriver.exe'
START = 'https://id.wiktionary.org/wiki/Kategori:id:Nomina_(dasar)'

BASE = 'https://id.wiktionary.org/wiki/Wiktionary:ProyekWiki_bahasa_Indonesia/Daftar_kata/Imbuhan/'
IMBUHAN = ['-kan', 'ber-', 'di-', 'me-', 'per-', 'ter-']

driver = webdriver.Chrome(PATH)
# driver.get(START)

# function to get list of kata dasar
def get_kata_dasar(element):
    ul = element.find_element_by_tag_name('ul')
    li = ul.find_elements_by_tag_name('li')
    words = [ a.find_element_by_tag_name('a').text for a in li ]

    return words

# function to get list of kata berimbuhan
def get_kata_berimbuhan(element):
    paragraphs = element.find_elements_by_tag_name('p')
    words = []
    for par in paragraphs:
        a = par.find_elements_by_tag_name('a')
        tmp = [ x.text for x in a ]
        words = words + tmp

    return words

# function to write list of words
def write_words(list_freq, output_file):
    print("Writing words to file...")
    data = open(output_file, 'a', encoding='utf-8')

    for word in list_freq:
        data.write(word + "\n")
    
    data.close()
    print("Finish writing...")

# scraping kata dasar
# words = []
# element = driver.find_element_by_class_name('mw-category-group')
# tmp = get_kata_dasar(element)
# words = words + tmp
# print(len(words))
# link = driver.find_element_by_link_text('halaman selanjutnya')
# link.click()

# try:
#     for i in range(0, 199):
#         element = WebDriverWait(driver, 10).until(
#             EC.presence_of_all_elements_located((By.CLASS_NAME, 'mw-category-group'))
#         )
#         tmp = get_kata_dasar(element[0])
#         words = words + tmp
#         print("Page ", i)
#         print(len(words))
#         try:
#             link = driver.find_element_by_link_text('halaman selanjutnya')
#             link.click()
#         except:
#             break
# finally:
#     driver.quit()
#     pass

# scraping kata berimbuhan
url = 'ter-'
print('Imbuhan ', url)

driver.get(BASE+url)
element = driver.find_element_by_class_name('mw-parser-output')
kata_berimbuhan = get_kata_berimbuhan(element)
print(len(kata_berimbuhan))
driver.quit()

print("Total words: ", len(kata_berimbuhan))

# write to text file
filename = '../data/id_wiki_list.txt'
write_words(kata_berimbuhan, filename)