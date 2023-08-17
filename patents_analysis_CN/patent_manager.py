import os
import re
import PyPDF2
import atexit
import logging
import numpy as np
from tqdm import tqdm


class patent_manager:
    """
    专利管理类,拥有诸多专利管理功能
    import patent.patent_manager as pm

    p_manager = pm.patent_manager()
    if p_manager.check_buffer():        # buffer加载比较快，毫秒级别响应，但是分析比较慢，如果存在buffer并想继续时，检测存在，进行process则进行继续分析
        # p_manager.append_patent_path(path) # 如果想在当前buffer中继续添加专利文本并分析，则使用append_buffer方法继续分析
        p_manager.process()
    else:                               # 新的加载专利并分析项目
        p_manager.load_patent_path(r"G:\back").process()

    p_manager.get(name)                 # 获取指定专利名称的专利信息
    p_manager.get_classify()            # 获取指定分类的专利信息
    """
    data_update_flag = False

    def __init__(self):
        logging.basicConfig(format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
                            level=logging.INFO)
        self.self_path = os.path.dirname(os.path.abspath(__file__))
        self.patent_path = []  # 专利路径
        self.patents = []  # 专利信息
        self.patents_error = []  # 分析错误的专利信息
        self.load_buffer()  # 加载缓存

    def __len__(self):
        return len(self.patents)

    def load_patent_path(self, root):
        """
        初始化专利路径，如果有临时路径的时候，给出提示并加载，没有的话，可以继续添加路径。
        :return: 专利目录
        """
        logging.info("专利目录加载中")
        if len(self.patent_path) == 0:
            self.append_path(root)
        else:
            logging.error("检测到当前环境有缓存，如果仍要继续执行，请先执行clear()方法清除缓存")
            raise Exception("buffer", "缓存未清理")
        logging.info("专利目录加载结束")
        return self

    def append_patent_path(self, root):
        """
        在现有的基础上添加
        :param root:
        :return:
        """
        logging.info("专利目录加载中")
        if len(self.patent_path) != 0:
            self.append_path(root)
        else:
            logging.error("当前并不存在缓存，请直接使用load_patent_path进行加载")
            raise Exception("buffer", "当前并不存在缓存")
        logging.info("专利目录加载结束")
        return self

    def append_path(self, root):
        """
        追加path数据
        :param root:
        :return:
        """
        patent_manager.data_update_flag = True
        for root, _, files in os.walk(root, topdown=False):
            for name in files:
                if not name.startswith(".") and name.endswith(".pdf"):
                    logging.debug(os.path.join(root, name))
                    self.patent_path.append({"path": os.path.join(root, name), "loaded": False})

    def process(self):
        """
        分析专利文本文件
        :return: 专利文件文本数组
        """
        patent_manager.data_update_flag = True
        for item in tqdm(self.patent_path):
            if item['loaded'] is False and item['path'] not in self.patents_error:
                try:
                    logging.debug("解析：{}".format(item['path']))
                    if item['path'].startswith(r"G:\back\CN113113181B_一种水下发光线缆.pdf"):
                        print("-----------")
                    self.patents.append(patent_analysis(item['path']).analysis())
                    item['loaded'] = True
                except Exception as e:
                    logging.error("解析失败：{}".format(item['path']))
                    self.patents_error.append(item['path'])

    def load_buffer(self):
        """
        记载所有缓存数据
        :return:
        """
        if self.check_buffer():
            logging.info("当前环境拥有缓存，正在加载缓存信息")
            self.patent_path = np.load(os.path.join(self.self_path, "patent_path.npy"), allow_pickle=True).tolist()
            self.patents = np.load(os.path.join(self.self_path, "patents.npy"), allow_pickle=True).tolist()
            if os.path.exists(os.path.join(self.self_path, "patents_error.npy")):
                self.patents_error = np.load(os.path.join(self.self_path, "patents_error.npy")).tolist()
            logging.info("缓存加载完成")
        else:
            logging.info("当前环境无缓存信息")
        return self

    def clear(self):
        """
        清除所有缓存数据
        :return:
        """
        patent_manager.data_update_flag = True
        self.patent_path = []
        self.patents = []
        self.patents_error = []
        if os.path.exists(os.path.join(self.self_path, "patent_path.npy")):
            os.remove(os.path.join(self.self_path, "patent_path.npy"))
        if os.path.exists(os.path.join(self.self_path, "patents.npy")):
            os.remove(os.path.join(self.self_path, "patents.npy"))
        if os.path.exists(os.path.join(self.self_path, "patents_error.npy")):
            os.remove(os.path.join(self.self_path, "patents_error.npy"))

    def check_buffer(self):
        """
        检测buff是否存在
        :return:
        """
        if os.path.exists(os.path.join(self.self_path, "patent_path.npy")) or os.path.exists(
                os.path.join(self.self_path, "patents.npy")):
            if os.path.exists(os.path.join(self.self_path, "patent_path.npy")) and os.path.exists(
                    os.path.join(self.self_path, "patents.npy")):
                return True
            else:
                raise Exception("buffer", "缓存文件异常")
        else:
            return False

    def items(self):
        """
        给定索引返回指定专利数据
        :param index:
        :return:
        """
        return self.patents


class patent_analysis:
    """
    专利分析功能的中间件
    """

    def __init__(self, path):
        """
        获取需要专利分析的文件路径
        cover、power、menu、img是专利说明书的四大组成部分，都独立成页，cover是单独一页。
        :param path:
        """
        self.cover = None
        self.power = []
        self.menu = []
        self.img = []
        self.patent = patent_info()
        self.info = {'power': 0, 'menu': 0, 'img': 0}
        self.patent_reader = PyPDF2.PdfFileReader(open(path, "rb"))

    def analysis(self):
        """
        将专利分析后生成专利信息，并返回
        :return:
        """
        self.pages_divide()         # 将页面进行分类
        self.extract_cover()        # 提取封面，封面中包含了诸多信息，提取以后进行数据检查，防止提取的出现问题
        self.check_divided()        # 检查当前提取数据是否正确，出现问题则报错
        self.extract_power()        # 提取权力要求书
        self.extract_menu()         # 提取说明书
        if self.patent.check_struct():
            return self.patent
        else:
            raise Exception("patent_info", '专利文本结构自检出现问题')

    def pages_divide(self):
        """
        将页面分为：首页、权利要求书、说明书、附图
        :return:
        """
        for page in self.patent_reader.pages:
            page_txt = page.extract_text()
            if re.match(r"(\(19\))(([\u4e00-\u9fa5])*( )*(\u3000)*)*国家知识产权局", page_txt):                                                      # 封面页面判定
                self.cover = page_txt
            elif re.search("权( )*(\u3000)*利( )*(\u3000)*要( )*(\u3000)*求( )*(\u3000)*书( )*(\u3000)*[0-9]*/[0-9]*( )*(\u3000)*页", page_txt):     # 权利要求书判定
                self.power.append(page_txt)
            elif re.search("说( )*(\u3000)*明( )*(\u3000)*书( )*(\u3000)*[0-9]*/[0-9]*( )*(\u3000)*页", page_txt):                                  # 说明书判定
                self.menu.append(page_txt)
            elif re.search("说( )*(\u3000)*明( )*(\u3000)*书( )*(\u3000)*附( )*(\u3000)*图( )*(\u3000)*[0-9]*/[0-9]*( )*(\u3000)*页", page_txt):     # 附图判定
                self.img.append(page_txt)

    def check_divided(self):
        if int(self.info.get('power')) == len(self.power) and int(self.info.get('menu')) == len(self.menu) and int(self.info.get('img')) == len(self.img):
            return True
        else:
            raise Exception("patent_info", '专利文本结构自检出现问题')

    def extract_cover(self):
        """
        提取封面的详细信息，记得每次提取都要检查多次
        :return:
        """
        # 提取页面数量，用两种方法进行提取
        struct = re.findall('权利要求书([0-9]+)页 *说明书([0-9]+)页 *附图([0-9]+)页', self.cover)
        if len(struct) == 3:
            self.info['power'], self.info['menu'], self.info['img'] = struct
            self.cover = self.cover[:re.search("([0-9]*)页", self.cover).start()-6]
        else:
            struct = re.findall("([0-9]*)页", self.cover.split("\n")[-5])
            if len(struct) == 3:
                self.info['power'], self.info['menu'], self.info['img'] = struct
            elif len(struct) == 2:
                self.info['power'], self.info['menu'] = struct
            else:
                raise Exception("analysis", "专利文本信息结构解析失败")
            self.cover = "".join(self.cover.split("\n")[:-5])
        # 按结构提取专利信息
        lines = re.findall("\([0-9]+\)[^\()]+", self.cover)
        if len(lines) >= 13:
            for line in lines:
                # 提取标题
                if line.startswith(r'(54)'):
                    line = line.replace(" ", "").replace("\n", "")
                    self.patent.title = line[8:]                                    # 这种方法比较稳定，防止因为\n有时存在，有时不存在而出现问题。
                # 提取分类
                elif line.startswith(r'(51)'):
                    line = line.replace(" ", "").replace("\n", "")
                    self.patent.classify = line[11:]
                # 提发明人
                elif line.startswith(r'(72)'):
                    self.patent.author = re.split('\u3000+| +', line[7:].strip())
                # 专利摘要
                elif line.startswith(r'(57)'):
                    self.patent.abs = line[6:].replace("\n", "").replace(" ", "")
        else:
            raise Exception("analysis", "专利文本封面解析失败")

    def extract_power(self):
        """
        提取权力要求
        :return:
        """
        # 提取所有权力要求书
        power = []
        for power_item in self.power:
            item = power_item[:re.search('权( )*(\u3000)*利( )*(\u3000)*要( )*(\u3000)*求( )*(\u3000)*书( )*(\u3000)*[0-9]*/[0-9]*( )*(\u3000)*页', power_item).start()].replace("\n", "").replace(" ", "").replace(" ", "")
            item = re.sub("\([0-9]*\)", "", item)
            power.append(re.sub("\（[0-9]*\）", "", item))
        self.patent.power = re.findall('([1-9]+\.[\u4e00-\u9fa5].*?)(?=[1-9]+\.[\u4e00-\u9fa5]|$)', "".join(power))

    def extract_menu(self):
        """
        提取说明书
        :return:
        """
        menu = []
        for menu_item in self.menu:
            menu.append(menu_item[:re.search('说( )*(\u3000)*明( )*(\u3000)*书( )*(\u3000)*[0-9]*/[0-9]*( )*(\u3000)*页',menu_item).start()])
        all = ''.join(menu)

        # 提取技术领域
        area_span = re.search(r'(?:技术领域 *[:：【；;]?)\s*([\s\S]*?)(?=(?:背景技术|发明内容|附图说明|具体实施方式|实施方式|具体实施)[:：【；;]?|$)', all)
        if area_span and len(area_span.span()) == 2:
            area = all[area_span.span()[0]:area_span.span()[1]]
            self.patent.area = re.sub('\[\d{4}\]', '', area.replace('\n', '').replace(' ', '')[10:])
        else:
            raise Exception('analysis', '技术领域提取失败')

        # 提取背景技术
        background_span = re.search(r'(?:背景技术 *[:：【；;]?)\s*([\s\S]*?)(?=(?:技术领域|发明内容|附图说明|具体实施方式|实施方式|具体实施)[:：【；;]?|$)', all)
        if background_span and len(background_span.span()) == 2:
            background = all[background_span.span()[0]:background_span.span()[1]]
            self.patent.background = re.sub('\[\d{4}\]', '', background.replace('\n', '').replace(' ', '')[10:])
        else:
            raise Exception('analysis', '背景技术提取失败')

        # 提取发明内容
        invent_span = re.search(r'(?:发明内容 *[:：【；;]?)\s*([\s\S]*?)(?=(?:技术领域|背景技术|附图说明|具体实施方式|实施方式|具体实施)[:：【；;]?|$)', all)
        if invent_span and len(invent_span.span()) == 2:
            invent = all[invent_span.span()[0]:invent_span.span()[1]]
            self.patent.invent = re.sub('\[\d{4}\]', '', invent.replace('\n', '').replace(' ', '')[10:])
        else:
            raise Exception('analysis', '发明内容提取失败')

        # 提取附图说明
        # diagram_span = re.search(r'(?:附图说明[:：【]?)\s*([\s\S]*?)(?=(?:技术领域|背景技术|发明内容|具体实施方式)[:：【]?|$)', all).span()
        # if len(diagram_span) == 2:
        #     self.patent.diagram = all[diagram_span[0]:diagram_span[1]]
        # else:
        #     raise Exception('analysis', '背景技术提取失败')

        # 提取具体实施方式
        embodiment_span = re.search(r'(?:(具体实施方式|实施方式|具体实施) *[:：【；;]?)\s*([\s\S]*?)(?=(?:技术领域|背景技术|发明内容|附图说明)[:：【；;]?|$)', all)
        if embodiment_span and len(embodiment_span.span()) == 2:
            embodiment = all[embodiment_span.span()[0]:embodiment_span.span()[1]]
            self.patent.embodiment = re.sub('\[\d{4}\]', '', embodiment.replace('\n', '').replace(' ', '')[12:])
        else:
            raise Exception('analysis', '具体实施方式提取失败')


class patent_info:
    """
    每一个类包含了诸多专利信息
    # patent_info 信息是通过patent_manager 获取所得到
    pi.get_abs              # 获取摘要
    pi.get_title            # 获取专利名称
    pi.author               # 作者
    pi.classify             # 分类结果
    pi.first_power          # 首项权力要求
    pi.power                # 权力要求书
    pi.area                 # 专利领域
    pi.background           # 专利背景
    pi.invent               # 发明内容
    pi.embodiment           # 具体实施方式
    pi.all                  # 全部内容
    pi.check("", tag="")    # 对比文档是否一致
    """

    ABS = 1
    TITLE = 2
    AUTHOR = 3
    CLASSIFY = 4
    FIRST_POWER = 5
    POWER = 6
    AREA = 7
    BACKGROUND = 8
    INVENT = 9
    EMBODIMENT = 10
    ALL = 11

    def __init__(self):
        """
        初始化专利信息
        :param abs:
        :param title:
        :param author:
        :param classify:
        :param first_power:
        :param power:
        :param area:
        :param background:
        :param invent:
        :param embodiment:
        """
        self.abs = ''
        self.title = ''
        self.author = ''
        self.classify = ''
        self.first_power = ''
        self.power = ''
        self.area = ''
        self.background = ''
        self.invent = ''
        self.embodiment = ''

    def all(self):
        """
        返回专利所有信息
        :return:
        """
        return self.abs + ";" + self.title + ";" + self.author + ";" + self.classify + ";" + self.power + ";" + self.area + ";" + self.background + ";" + self.invent + ";" + self.embodiment

    def get(self, tag):
        """
        获取专利信息
        :param line:
        :param tag:
        :return:
        """
        if tag == self.ABS:
            return self.abs
        elif tag == self.TITLE:
            return self.title
        elif tag == self.AUTHOR:
            return self.author
        elif tag == self.CLASSIFY:
            return self.classify
        elif tag == self.FIRST_POWER:
            return self.first_power
        elif tag == self.POWER:
            return self.POWER
        elif tag == self.AREA:
            return self.area
        elif tag == self.BACKGROUND:
            return self.background
        elif tag == self.INVENT:
            return self.invent
        elif tag == self.EMBODIMENT:
            return self.embodiment

    def check_struct(self):
        return bool(re.match('[A-Z]{1}\d{2}[A-Z]{1}\d{1,2}/\d{1,3}$', self.classify))


pm = patent_manager()


@staticmethod
@atexit.register
def exit():
    """
    结束时执行的代码
    :return:
    """
    if save_data():
        logging.error("异常退出，进程数据已保存")


def save_data():
    """
    保存pm所有数据
    :return:
    """
    if pm.data_update_flag and len(pm.patents) > 0 :
        np.save(os.path.join(pm.self_path, "patent_path.npy"), pm.patent_path)
        np.save(os.path.join(pm.self_path, "patents.npy"), pm.patents)
        np.save(os.path.join(pm.self_path, "patents_error.npy"), pm.patents_error)
        return True
    return False
