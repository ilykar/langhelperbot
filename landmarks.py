# landmarks.py - поиск информации о достопримечательностях с поддержкой разных языков

import requests
import logging
import re
from urllib.parse import quote

logger = logging.getLogger(__name__)

# Расширенный словарь достопримечательностей на РУССКОМ языке (ключи)
LANDMARKS_RU = {
    # Россия
    'красная площадь': {
        'name': 'Красная площадь',
        'description': 'Москва, Россия - главная площадь страны',
        'en_name': 'Red Square'
    },
    'кремль': {
        'name': 'Кремль',
        'description': 'Москва, историческая крепость',
        'en_name': 'Kremlin'
    },
    'эрмитаж': {
        'name': 'Эрмитаж',
        'description': 'Санкт-Петербург, музей искусств',
        'en_name': 'Hermitage Museum'
    },
    'петергоф': {
        'name': 'Петергоф',
        'description': 'Санкт-Петербург, дворцово-парковый ансамбль',
        'en_name': 'Peterhof Palace'
    },
    'собор василия блаженного': {
        'name': 'Собор Василия Блаженного',
        'description': 'Москва, православный храм на Красной площади',
        'en_name': 'Saint Basil\'s Cathedral'
    },
    'мавзолей ленина': {
        'name': 'Мавзолей Ленина',
        'description': 'Москва, усыпальница на Красной площади',
        'en_name': 'Lenin\'s Mausoleum'
    },
    'третьяковская галерея': {
        'name': 'Третьяковская галерея',
        'description': 'Москва, музей русского искусства',
        'en_name': 'Tretyakov Gallery'
    },
    'большой театр': {
        'name': 'Большой театр',
        'description': 'Москва, театр оперы и балета',
        'en_name': 'Bolshoi Theatre'
    },
    
    # Европа
    'эйфелева башня': {
        'name': 'Эйфелева башня',
        'description': 'Париж, Франция - металлическая башня',
        'en_name': 'Eiffel Tower'
    },
    'лувр': {
        'name': 'Лувр',
        'description': 'Париж, Франция - художественный музей',
        'en_name': 'Louvre Museum'
    },
    'колизей': {
        'name': 'Колизей',
        'description': 'Рим, Италия - амфитеатр',
        'en_name': 'Colosseum'
    },
    'биг бен': {
        'name': 'Биг-Бен',
        'description': 'Лондон, Великобритания - часовая башня',
        'en_name': 'Big Ben'
    },
    'статуя свободы': {
        'name': 'Статуя Свободы',
        'description': 'Нью-Йорк, США - символ свободы',
        'en_name': 'Statue of Liberty'
    },
    'великая китайская стена': {
        'name': 'Великая Китайская стена',
        'description': 'Китай - оборонительное сооружение',
        'en_name': 'Great Wall of China'
    },
    'тадж махал': {
        'name': 'Тадж-Махал',
        'description': 'Индия - мавзолей-мечеть',
        'en_name': 'Taj Mahal'
    },
    
    # Азия
    'фудзияма': {
        'name': 'Фудзияма',
        'description': 'Япония - вулкан и священная гора',
        'en_name': 'Mount Fuji'
    },
    'ангкор ват': {
        'name': 'Ангкор-Ват',
        'description': 'Камбоджа - храмовый комплекс',
        'en_name': 'Angkor Wat'
    },
    
    # Для нейросети (новые)
    'архитектурный объект': {
        'name': 'Архитектурный объект',
        'description': 'Обнаруженный архитектурный объект',
        'en_name': 'Architectural Object'
    },
    'замок': {
        'name': 'Замок',
        'description': 'Историческое оборонительное сооружение',
        'en_name': 'Castle'
    },
    'церковь': {
        'name': 'Церковь',
        'description': 'Религиозное сооружение',
        'en_name': 'Church'
    },
    'мечеть': {
        'name': 'Мечеть',
        'description': 'Мусульманское молитвенное сооружение',
        'en_name': 'Mosque'
    },
    'дворец': {
        'name': 'Дворец',
        'description': 'Парадное здание для знати',
        'en_name': 'Palace'
    },
    'мост': {
        'name': 'Мост',
        'description': 'Инженерное сооружение для преодоления препятствий',
        'en_name': 'Bridge'
    },
    'небоскрёб': {
        'name': 'Небоскрёб',
        'description': 'Высотное здание',
        'en_name': 'Skyscraper'
    },
    'музей': {
        'name': 'Музей',
        'description': 'Учреждение для хранения и展示 экспонатов',
        'en_name': 'Museum'
    },
    'стадион': {
        'name': 'Стадион',
        'description': 'Спортивное сооружение',
        'en_name': 'Stadium'
    },
    'башня': {
        'name': 'Башня',
        'description': 'Высокое сооружение',
        'en_name': 'Tower'
    },
}

# Словарь на АНГЛИЙСКОМ языке
LANDMARKS_EN = {
    'eiffel tower': {'ru_name': 'Эйфелева башня', 'en_name': 'Eiffel Tower'},
    'red square': {'ru_name': 'Красная площадь', 'en_name': 'Red Square'},
    'big ben': {'ru_name': 'Биг-Бен', 'en_name': 'Big Ben'},
    'statue of liberty': {'ru_name': 'Статуя Свободы', 'en_name': 'Statue of Liberty'},
    'colosseum': {'ru_name': 'Колизей', 'en_name': 'Colosseum'},
    'kremlin': {'ru_name': 'Кремль', 'en_name': 'Kremlin'},
    'taj mahal': {'ru_name': 'Тадж-Махал', 'en_name': 'Taj Mahal'},
    'great wall of china': {'ru_name': 'Великая Китайская стена', 'en_name': 'Great Wall of China'},
    'mount fuji': {'ru_name': 'Фудзияма', 'en_name': 'Mount Fuji'},
    'angkor wat': {'ru_name': 'Ангкор-Ват', 'en_name': 'Angkor Wat'},
    'louvre': {'ru_name': 'Лувр', 'en_name': 'Louvre Museum'},
    'hermitage': {'ru_name': 'Эрмитаж', 'en_name': 'Hermitage Museum'},
    'saint basil\'s cathedral': {'ru_name': 'Собор Василия Блаженного', 'en_name': 'Saint Basil\'s Cathedral'},
    'peterhof': {'ru_name': 'Петергоф', 'en_name': 'Peterhof Palace'},
    'castle': {'ru_name': 'Замок', 'en_name': 'Castle'},
    'church': {'ru_name': 'Церковь', 'en_name': 'Church'},
    'mosque': {'ru_name': 'Мечеть', 'en_name': 'Mosque'},
    'palace': {'ru_name': 'Дворец', 'en_name': 'Palace'},
    'bridge': {'ru_name': 'Мост', 'en_name': 'Bridge'},
    'skyscraper': {'ru_name': 'Небоскрёб', 'en_name': 'Skyscraper'},
    'museum': {'ru_name': 'Музей', 'en_name': 'Museum'},
    'stadium': {'ru_name': 'Стадион', 'en_name': 'Stadium'},
    'tower': {'ru_name': 'Башня', 'en_name': 'Tower'},
}

# Интересные факты
LANDMARK_FACTS = {
    'красная площадь': 'Интересный факт: Изначально называлась "Торг", а современное название получила в 17 веке',
    'кремль': 'Интересный факт: В Кремле 20 башен, каждая имеет свое название и историю',
    'эйфелева башня': 'Интересный факт: Построена за 2 года и 2 месяца, изначально планировалась как временное сооружение',
    'колизей': 'Интересный факт: Вмещал до 50 000 зрителей, имел раздвижную крышу',
    'биг бен': 'Интересный факт: Название относится не к башне, а к 13-тонному колоколу внутри',
    'статуя свободы': 'Интересный факт: Подарок Франции США к 100-летию независимости',
    'великая китайская стена': 'Интересный факт: Ее длина составляет около 21 196 км',
    'тадж махал': 'Интересный факт: Строился 22 года, для его отделки использовались полудрагоценные камни',
    'эрмитаж': 'Интересный факт: Чтобы обойти все экспозиции, потребуется пройти 24 км',
    'петергоф': 'Интересный факт: Имеет 176 фонтанов и 4 каскада',
    'лувр': 'Интересный факт: Самый посещаемый музей в мире, открыт в 1793 году',
    'фудзияма': 'Интересный факт: Активный вулкан, последнее извержение было в 1707 году',
    'ангкор ват': 'Интересный факт: Крупнейший религиозный памятник в мире',
    'собор василия блаженного': 'Интересный факт: Построен в 1555-1561 годах по приказу Ивана Грозного',
    'третьяковская галерея': 'Интересный факт: Основана в 1856 году купцом Павлом Третьяковым',
    'большой театр': 'Интересный факт: Открыт в 1825 году, пострадал от нескольких пожаров',
    
    # Для нейросети
    'архитектурный объект': 'Интересный факт: Обнаружено нейросетью PyTorch с использованием компьютерного зрения',
    'замок': 'Интересный факт: Замки строились для защиты от нападений, часто на холмах',
    'церковь': 'Интересный факт: Архитектура церквей часто символична и отражает религиозные традиции',
    'мечеть': 'Интересный факт: Мечети часто ориентированы в сторону Мекки (кибла)',
    'дворец': 'Интересный факт: Дворцы демонстрировали богатство и власть их владельцев',
    'мост': 'Интересный факт: Мосты существуют с древних времен для преодоления водных преград',
    'небоскрёб': 'Интересный факт: Первым небоскребом считается Home Insurance Building в Чикаго (1885)',
    'музей': 'Интересный факт: Слово "музей" происходит от греческого "мусейон" - храм муз',
    'стадион': 'Интересный факт: Самый большой стадион в мире - Стадион Первого мая в Пхеньяне (150 000 мест)',
    'башня': 'Интересный факт: Башни строили для обзора местности, связи или как символы',
}

# Синонимы для более гибкого поиска
SYNONYMS = {
    'московский кремль': 'кремль',
    'кремль в москве': 'кремль',
    'the kremlin': 'кремль',
    'парижская башня': 'эйфелева башня',
    'eiffel': 'эйфелева башня',
    'лондонская башня': 'биг бен',
    'лондонский биг бен': 'биг бен',
    'bigben': 'биг бен',
    'великая стена': 'великая китайская стена',
    'great wall': 'великая китайская стена',
    'тадж': 'тадж махал',
    'taj': 'тадж махал',
    'гора фудзи': 'фудзияма',
    'fuji': 'фудзияма',
    'ангкор': 'ангкор ват',
}

def find_landmark_info(text):
    """Поиск достопримечательности в тексте на русском или английском"""
    text_lower = text.lower().strip()
    
    # 0. Сначала проверяем синонимы
    for synonym, main_name in SYNONYMS.items():
        if synonym in text_lower:
            text_lower = text_lower.replace(synonym, main_name)
    
    # 1. Сначала ищем на русском
    for landmark_key, info in LANDMARKS_RU.items():
        if landmark_key in text_lower:
            fact = LANDMARK_FACTS.get(landmark_key, 'Интересный факт: Эта достопримечательность имеет богатую историю')
            return {
                'found': True,
                'name': info['name'],
                'description': info['description'],
                'fact': fact,
                'en_name': info['en_name']
            }
    
    # 2. Ищем на английском
    for landmark_key, info in LANDMARKS_EN.items():
        if landmark_key in text_lower:
            # Находим русский аналог
            ru_name = info['ru_name']
            ru_key = ru_name.lower()
            
            if ru_key in LANDMARKS_RU:
                fact = LANDMARK_FACTS.get(ru_key, 'Интересный факт: Эта достопримечательность имеет богатую историю')
                return {
                    'found': True,
                    'name': ru_name,
                    'description': LANDMARKS_RU[ru_key]['description'],
                    'fact': fact,
                    'en_name': info['en_name']
                }
    
    # 3. Попробуем поискать отдельные слова
    words = re.findall(r'\b\w+\b', text_lower)
    for word in words:
        if len(word) > 3:  # Ищем слова длиннее 3 букв
            for landmark_key, info in LANDMARKS_RU.items():
                if word in landmark_key:
                    fact = LANDMARK_FACTS.get(landmark_key, 'Интересный факт: Эта достопримечательность имеет богатую историю')
                    return {
                        'found': True,
                        'name': info['name'],
                        'description': info['description'],
                        'fact': fact,
                        'en_name': info['en_name']
                    }
    
    return {'found': False}

def search_wikipedia(landmark_name, lang='ru'):
    """Поиск в Википедии через API"""
    try:
        # Кодируем название для URL
        encoded_name = quote(landmark_name)
        
        if lang == 'ru':
            url = f"https://ru.wikipedia.org/api/rest_v1/page/summary/{encoded_name}"
        else:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_name}"
        
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'description': data.get('extract', 'Описание не найдено'),
                'url': data.get('content_urls', {}).get('desktop', {}).get('page', '')
            }
    except Exception as e:
        logger.error(f"Ошибка поиска в Wikipedia: {e}")
    
    return None

def get_landmark_by_english_name(en_name):
    """Получить информацию по английскому названию"""
    en_name_lower = en_name.lower().strip()
    
    # Ищем в английском словаре
    for landmark_key, info in LANDMARKS_EN.items():
        if landmark_key == en_name_lower:
            ru_name = info['ru_name']
            ru_key = ru_name.lower()
            
            if ru_key in LANDMARKS_RU:
                fact = LANDMARK_FACTS.get(ru_key, 'Интересный факт: Эта достопримечательность имеет богатую историю')
                return {
                    'found': True,
                    'name': ru_name,
                    'description': LANDMARKS_RU[ru_key]['description'],
                    'fact': fact,
                    'en_name': info['en_name']
                }
    
    return {'found': False}

# Дополнительная функция для нейросети
def get_landmark_by_type(landmark_type):
    """Получить информацию по типу достопримечательности (для нейросети)"""
    landmark_type = landmark_type.lower()
    
    if landmark_type in LANDMARKS_RU:
        info = LANDMARKS_RU[landmark_type]
        fact = LANDMARK_FACTS.get(landmark_type, 'Интересный факт: Эта достопримечательность имеет богатую историю')
        
        return {
            'found': True,
            'name': info['name'],
            'description': info['description'],
            'fact': fact,
            'en_name': info['en_name']
        }
    
    return {'found': False}