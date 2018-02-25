import requests
from lxml import etree

#Some User Agents
hds=[{'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'},\
{'User-Agent':'Mozilla/5.0 (Windows NT 6.2) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.12 Safari/535.11'},\
{'User-Agent': 'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Trident/6.0)'}]

with open('douban_book_top250.txt', 'w', encoding='utf') as f:
	for i in range(0,226,25):
		url = 'https://book.douban.com/top250?start={}'.format(i)
		data = requests.get(url).text

		page = etree.HTML(data)
		file = page.xpath('//*[@id="content"]/div/div[1]/div/table')
		for div in file:
			title = div.xpath('./tr/td[2]/div[1]/a/@title')
			pf = div.xpath('./tr/td[2]/div[2]/span[2]/text()')
			words = div.xpath('./tr/td[2]/p[2]/span/text()')

			f.write("{}{}{}\n".format(title,pf,words))

print("done")