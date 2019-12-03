import requests
import os
from bs4 import BeautifulSoup

class Crawler_ :
    def __init__(self, start, end, dir = 'yumi_cell'):
        self.URL_LIST = ['https://m.comic.naver.com/webtoon/detail.nhn?titleId=651673&no=' + str(i) for i in
                    range(start, end + 1)]
        # 이게 오프닝부터 238화 까지임(이상하게 중간에 날아간 인덱스들이 있음)
        # 233화(인덱스로 234)부터 계속 중복되는 타이틀 이미지가 나오니 제거해줘야함
        self.title = dir

        self.cnt = 0 #이미지 번호

    def crawl_(self, epi):
        epi_url = self.URL_LIST[epi]
        html = requests.get(epi_url).text
        soup = BeautifulSoup(html, 'html.parser')

        # title_ep = soup.find('meta', property="og:title").get('content')
        # title = title_ep[:title_ep.find('-')].strip()
        category = 'raw'

        cut_list = soup.find_all('img', class_='swiper-lazy', alt="")
        # 233화(인덱스로 234)부터 계속 중복되는 타이틀 이미지가 나오니 제거해줘야함
        if epi >= 234:
            cut_list = cut_list[1:]
        for img_tag in cut_list:
            image_file_url = img_tag['data-src']
            image_dir_path = os.path.join(os.path.dirname(__file__), self.title, category)
            image_file_path = os.path.join(image_dir_path, "%5d.jpg "%self.cnt)

            if not os.path.exists(image_dir_path):
                os.makedirs(image_dir_path)

            # print(image_file_path)

            headers = {'Referer': epi_url}
            image_file_data = requests.get(image_file_url, headers=headers).content
            open(image_file_path, 'wb').write(image_file_data)
            self.cnt += 1

    def run(self):
        for i in range(len(self.URL_LIST)):
            self.crawl_(i)

        print('Completed !')

# C = Crawler_(1, 241)
# C.run()
# C = Crawler_(242, 342, 'test') #테스트용
# C.run()
C = Crawler_(242, 439, 'more_data') #테스트용
C.run()