import telebot
from googletrans import Translator
import easyocr
import cv2
import numpy as np
from PIL import Image
import io
import sqlite3
from datetime import datetime
import os
from dotenv import load_dotenv
import logging

# ========== НАСТРОЙКА ЛОГГИРОВАНИЯ ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== ЗАГРУЗКА КОНФИГУРАЦИИ ==========
load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

if not TOKEN:
    print("❌ ОШИБКА: Токен не найден в .env файле!")
    print("Создайте файл .env с содержимым: BOT_TOKEN=ваш_токен")
    exit(1)

if ":" not in TOKEN:
    print("❌ ОШИБКА: Неверный формат токена!")
    print("Токен должен содержать двоеточие: числа:буквы")
    exit(1)

# ========== ИНИЦИАЛИЗАЦИЯ ==========
print("🚀 Запуск LangHelperBot с OCR...")
bot = telebot.TeleBot(TOKEN)
translator = Translator()

# Инициализация EasyOCR с правильными комбинациями языков
print("🔄 Инициализация нейросети EasyOCR...")
print("⚠️ Используется CPU. Это нормально для проекта.")

try:
    # Для EasyOCR некоторые языки требуют комбинации с английским
    # Создаем несколько читателей для разных языковых групп
    
    # 1. Основные европейские языки (могут работать вместе)
    print("📥 Загружаю европейские языки...")
    reader_europe = easyocr.Reader(['en', 'ru', 'de', 'fr', 'es'], gpu=False)
    
    # 2. Азиатские языки (требуют отдельной загрузки)
    print("📥 Загружаю японский язык...")
    reader_japanese = easyocr.Reader(['ja', 'en'], gpu=False)
    
    print("📥 Загружаю корейский язык...")
    reader_korean = easyocr.Reader(['ko', 'en'], gpu=False)
    
    # 3. Китайский (если нужен)
    try:
        print("📥 Загружаю китайский язык...")
        reader_chinese = easyocr.Reader(['ch_sim', 'en'], gpu=False)
        chinese_loaded = True
    except:
        print("⚠️ Китайский язык не загружен")
        chinese_loaded = False
    
    readers = {
        'europe': reader_europe,
        'japanese': reader_japanese,
        'korean': reader_korean
    }
    
    if chinese_loaded:
        readers['chinese'] = reader_chinese
    
    print("✅ Нейросеть EasyOCR успешно загружена")
    print(f"✅ Загружено языков: {len(readers)} групп")
    
except Exception as e:
    print(f"❌ Ошибка загрузки EasyOCR: {e}")
    print("Создаю упрощенного читателя только для английского...")
    
    # Создаем минимального читателя только для английского
    try:
        readers = {'english': easyocr.Reader(['en'], gpu=False)}
        print("✅ Загружен английский язык (минимальная версия)")
    except:
        print("❌ Критическая ошибка: не удалось загрузить OCR")
        print("Попробуйте: pip install torch==1.10.0 torchvision==0.11.0 --index-url https://download.pytorch.org/whl/cpu")
        exit(1)

# Константы
DB_FILE = "langhelper.db"

# ========== БАЗА ДАННЫХ ==========
def init_db():
    """Инициализация базы данных"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            target_language TEXT DEFAULT 'ru',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            type TEXT,
            original_text TEXT,
            translated_text TEXT,
            source_lang TEXT,
            target_lang TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ База данных инициализирована")
    except Exception as e:
        print(f"❌ Ошибка БД: {e}")

def add_user(user_id, username="", first_name=""):
    """Добавление пользователя"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT OR IGNORE INTO users (user_id, username, first_name) 
        VALUES (?, ?, ?)
        ''', (user_id, username, first_name))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Ошибка добавления пользователя: {e}")

def add_to_history(user_id, type_, original, translated, src_lang, target_lang):
    """Добавление в историю"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO history (user_id, type, original_text, translated_text, source_lang, target_lang)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, type_, original[:1000], translated[:1000], src_lang, target_lang))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Ошибка добавления в историю: {e}")

def get_user_language(user_id):
    """Получение языка пользователя"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('SELECT target_language FROM users WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else 'ru'
    except:
        return 'ru'

def set_user_language(user_id, lang):
    """Установка языка"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
        INSERT OR REPLACE INTO users (user_id, target_language) 
        VALUES (?, ?)
        ''', (user_id, lang))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Ошибка установки языка: {e}")

def get_user_history(user_id, limit=5):
    """Получение истории"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
        SELECT type, original_text, translated_text, source_lang, target_lang, timestamp
        FROM history 
        WHERE user_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
        ''', (user_id, limit))
        history = cursor.fetchall()
        conn.close()
        return history
    except:
        return []

def clear_user_history(user_id):
    """Очистка истории"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM history WHERE user_id = ?', (user_id,))
        conn.commit()
        conn.close()
        return True
    except:
        return False

# Инициализация БД
init_db()

# ========== КЛАВИАТУРЫ ==========
from telebot import types

def get_main_keyboard():
    """Главное меню"""
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.row("📸 Распознать фото", "📝 Переводчик")
    markup.row("🌍 Язык перевода", "📚 История")
    markup.row("❓ Помощь")
    return markup

def get_lang_keyboard():
    """Выбор языка"""
    markup = types.InlineKeyboardMarkup(row_width=3)
    languages = [
        ("🇬🇧 Английский", "en"),
        ("🇩🇪 Немецкий", "de"),
        ("🇫🇷 Французский", "fr"),
        ("🇪🇸 Испанский", "es"),
        ("🇯🇵 Японский", "ja"),
        ("🇰🇷 Корейский", "ko"),
        ("🇷🇺 Русский", "ru"),
        ("🇮🇹 Итальянский", "it"),
        ("🇵🇹 Португальский", "pt"),
        ("🇦🇪 Арабский", "ar"),
        ("🇹🇷 Турецкий", "tr"),
        ("🇨🇳 Китайский", "zh-cn")
    ]
    for name, code in languages:
        markup.add(types.InlineKeyboardButton(name, callback_data=f"lang_{code}"))
    return markup

# ========== OCR ФУНКЦИИ ==========
def process_image_ocr(image_bytes):
    """Обработка изображения и распознавание текста"""
    try:
        # Преобразуем bytes в изображение
        image = Image.open(io.BytesIO(image_bytes))
        
        # Конвертируем в numpy array для OpenCV
        img_np = np.array(image)
        
        # Конвертируем RGB в BGR если нужно
        if len(img_np.shape) == 3:
            if img_np.shape[2] == 4:  # RGBA
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        all_results = []
        
        # Пробуем всех читателей
        for reader_name, reader in readers.items():
            try:
                result = reader.readtext(img_np, detail=0, paragraph=True)
                if result:
                    text = ' '.join(result).strip()
                    if text and len(text) > 1:
                        all_results.append((reader_name, text))
            except Exception as e:
                logger.error(f"Ошибка OCR {reader_name}: {e}")
        
        # Если ничего не распознано, пробуем улучшить изображение
        if not all_results:
            # Конвертируем в grayscale
            if len(img_np.shape) == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_np
            
            # Улучшаем контраст
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Пробуем снова с улучшенным изображением
            for reader_name, reader in readers.items():
                try:
                    result = reader.readtext(enhanced, detail=0, paragraph=True)
                    if result:
                        text = ' '.join(result).strip()
                        if text and len(text) > 1:
                            all_results.append((f"{reader_name}_enhanced", text))
                except:
                    pass
        
        if not all_results:
            return None
        
        # Выбираем самый длинный результат
        best_result = max(all_results, key=lambda x: len(x[1]))
        return best_result[1]
        
    except Exception as e:
        logger.error(f"Общая ошибка OCR: {e}")
        return None

# ========== ОБРАБОТЧИКИ КОМАНД ==========
@bot.message_handler(commands=['start'])
def cmd_start(message):
    """Начало работы"""
    user_id = message.from_user.id
    username = message.from_user.username or ""
    first_name = message.from_user.first_name or ""
    
    add_user(user_id, username, first_name)
    set_user_language(user_id, 'ru')
    
    welcome = f"""
🤖 **Привет, {first_name}! Я LangHelperBot**

Я ваш помощник для путешествий и изучения языков с использованием нейросетей:

📸 **РАСПОЗНАВАНИЕ ТЕКСТА С ФОТО:**
• Сфотографируйте вывеску, меню, указатель
• Отправьте фото мне
• Нейросеть распознает текст
• Я переведу на нужный язык

📝 **ТЕКСТОВЫЙ ПЕРЕВОД:**
• Отправьте текст на любом языке
• Я определю язык и переведу

🌍 **ПОДДЕРЖИВАЕМЫЕ ЯЗЫКИ:**
• Распознавание: Английский, Русский, Немецкий, Французский, Испанский, Японский, Корейский
• Перевод: 100+ языков

⚙️ **ФУНКЦИИ:**
• Выбор языка перевода
• История всех переводов
• Автоопределение языка

**Просто отправьте мне фото с текстом или любой текст для перевода!**
    """
    
    bot.send_message(message.chat.id, welcome, 
                    reply_markup=get_main_keyboard(),
                    parse_mode='Markdown')

@bot.message_handler(commands=['help'])
def cmd_help(message):
    """Помощь"""
    help_text = """
📖 **КАК ИСПОЛЬЗОВАТЬ LANGHELPERBOT:**

📸 **ДЛЯ ФОТОГРАФИЙ:**
1. Сфотографируйте текст (вывеска, меню, книга, указатель)
2. Отправьте фото в этот чат
3. Нейросеть распознает текст
4. Я переведу на выбранный язык

📝 **ДЛЯ ТЕКСТА:**
1. Напишите текст на любом языке
2. Я определю язык автоматически
3. Переведу на нужный вам язык

⚙️ **КОМАНДЫ:**
/start - Начало работы
/help - Эта справка  
/language - Выбрать язык перевода
/history - История переводов
/clear - Очистить историю

🌍 **СОВЕТЫ ДЛЯ ЛУЧШЕГО РАСПОЗНАВАНИЯ:**
• Фотографируйте при хорошем освещении
• Текст должен быть четким и контрастным
• Избегайте сильных наклонов камеры
• Фотографируйте текст ровно

**ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ В ПУТЕШЕСТВИИ:**
• В ресторане: сфотографируйте меню
• На улице: сфотографируйте указатель или вывеску
• В музее: сфотографируйте описание экспоната
• В магазине: сфотографируйте этикетку товара
• На вокзале: сфотографируйте расписание
    """
    
    bot.send_message(message.chat.id, help_text, parse_mode='Markdown')

@bot.message_handler(commands=['language', 'lang'])
def cmd_language(message):
    """Выбор языка"""
    bot.send_message(message.chat.id, 
                    "🌍 **ВЫБЕРИТЕ ЯЗЫК ПЕРЕВОДА:**\n\n"
                    "На этот язык я буду переводить все тексты.",
                    reply_markup=get_lang_keyboard(),
                    parse_mode='Markdown')

@bot.message_handler(commands=['history'])
def cmd_history(message):
    """История переводов"""
    user_id = message.from_user.id
    history = get_user_history(user_id, 10)
    
    if not history:
        bot.send_message(message.chat.id, 
                        "📭 **ИСТОРИЯ ПЕРЕВОДОВ ПУСТА**\n\n"
                        "Сделайте первый перевод, отправив текст или фото!",
                        parse_mode='Markdown')
        return
    
    response = "📚 **ПОСЛЕДНИЕ ПЕРЕВОДЫ:**\n\n"
    
    for i, (type_, orig, trans, src, targ, time) in enumerate(history, 1):
        icon = "📸" if type_ == 'photo' else "📝"
        
        # Обрезаем длинный текст
        orig_display = orig[:50] + "..." if len(orig) > 50 else orig
        trans_display = trans[:50] + "..." if len(trans) > 50 else trans
        
        # Форматируем время
        try:
            time_str = datetime.strptime(time, "%Y-%m-%d %H:%M:%S").strftime("%d.%m %H:%M")
        except:
            time_str = time[:16]
        
        response += f"{icon} **{i}. {orig_display}**\n"
        response += f"   → {trans_display}\n"
        response += f"   [{src.upper()}→{targ.upper()}] {time_str}\n\n"
    
    bot.send_message(message.chat.id, response, parse_mode='Markdown')

@bot.message_handler(commands=['clear'])
def cmd_clear(message):
    """Очистка истории"""
    user_id = message.from_user.id
    if clear_user_history(user_id):
        bot.send_message(message.chat.id, "✅ История переводов очищена!")
    else:
        bot.send_message(message.chat.id, "❌ Ошибка при очистке истории")

@bot.message_handler(commands=['status'])
def cmd_status(message):
    """Статус бота"""
    status_text = """
🤖 **СТАТУС LANGHELPERBOT:**

✅ **Система:**
• Бот работает нормально
• База данных подключена
• Переводчик активен

🔧 **Техническая информация:**
• OCR работает на CPU (это нормально)
• Загружено языковых моделей: несколько
• Распознавание поддерживает: Английский, Русский, Немецкий, Французский, Испанский, Японский, Корейский
• Перевод поддерживает: 100+ языков

📊 **Производительность:**
• Распознавание фото: ~5-10 секунд
• Текстовый перевод: мгновенно
• История сохраняется в базу данных

💡 **Примечание:**
Предупреждение "Using CPU" означает, что нейросеть работает на процессоре, а не на видеокарте.
Это абсолютно нормально для учебного проекта!
    """
    
    bot.send_message(message.chat.id, status_text, parse_mode='Markdown')

# ========== ОБРАБОТКА ФОТО ==========
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    """Обработка фотографий"""
    user_id = message.from_user.id
    
    # Отправляем сообщение о начале обработки
    processing_msg = bot.send_message(message.chat.id, 
                                     "🔄 **ОБРАБАТЫВАЮ ФОТО...**\n"
                                     "Распознаю текст с помощью нейросети...",
                                     parse_mode='Markdown')
    
    try:
        # Получаем фото (самое высокое качество)
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Распознаем текст с фото
        bot.edit_message_text("🔍 **РАСПОЗНАЮ ТЕКСТ...**\n"
                             "Нейросеть анализирует изображение...",
                             message.chat.id,
                             processing_msg.message_id,
                             parse_mode='Markdown')
        
        recognized_text = process_image_ocr(downloaded_file)
        
        if not recognized_text or len(recognized_text.strip()) < 2:
            bot.edit_message_text("❌ **НЕ УДАЛОСЬ РАСПОЗНАТЬ ТЕКСТ**\n\n"
                                 "Возможные причины:\n"
                                 "• Текст на фото нечеткий\n"
                                 "• Слишком плохое освещение\n"
                                 "• Шрифт не поддерживается\n"
                                 "• Язык текста не поддерживается\n\n"
                                 "Попробуйте другое фото с более четким текстом.",
                                 message.chat.id,
                                 processing_msg.message_id,
                                 parse_mode='Markdown')
            return
        
        bot.edit_message_text("🌍 **ОПРЕДЕЛЯЮ ЯЗЫК И ПЕРЕВОДЖУ...**",
                             message.chat.id,
                             processing_msg.message_id,
                             parse_mode='Markdown')
        
        # Определяем язык текста
        try:
            detected = translator.detect(recognized_text)
            src_lang = detected.lang
            confidence = detected.confidence * 100
        except:
            src_lang = 'en'
            confidence = 0.0
        
        # Получаем язык пользователя
        target_lang = get_user_language(user_id)
        
        # Переводим текст
        try:
            translation = translator.translate(recognized_text, src=src_lang, dest=target_lang)
        except:
            # Если не удалось перевести, пробуем английский как источник
            translation = translator.translate(recognized_text, dest=target_lang)
            src_lang = 'en'
        
        # Сохраняем в историю
        add_to_history(user_id, 'photo', recognized_text, translation.text, src_lang, target_lang)
        
        # Формируем ответ
        lang_names = {
            'en': 'английский', 'ru': 'русский', 'de': 'немецкий',
            'fr': 'французский', 'es': 'испанский', 'ja': 'японский',
            'ko': 'корейский', 'it': 'итальянский', 'pt': 'португальский',
            'ar': 'арабский', 'tr': 'турецкий', 'zh-cn': 'китайский'
        }
        
        src_name = lang_names.get(src_lang, src_lang)
        targ_name = lang_names.get(target_lang, target_lang)
        
        # Обрезаем слишком длинный текст для отображения
        display_text = recognized_text[:400] + "..." if len(recognized_text) > 400 else recognized_text
        
        response = f"""
📸 **ТЕКСТ РАСПОЗНАН С ФОТО:**
`{display_text}`

🌍 **ОПРЕДЕЛЕН ЯЗЫК:** {src_name.upper()} (точность: {confidence:.1f}%)
🎯 **ПЕРЕВОД НА {targ_name.upper()}:**
{translation.text}
        """
        
        bot.edit_message_text(response,
                             message.chat.id,
                             processing_msg.message_id,
                             parse_mode='Markdown')
        
    except Exception as e:
        error_msg = f"❌ **ОШИБКА ПРИ ОБРАБОТКЕ ФОТО:**\n\n`{str(e)[:200]}`"
        bot.edit_message_text(error_msg,
                             message.chat.id,
                             processing_msg.message_id,
                             parse_mode='Markdown')

# ========== ОБРАБОТКА ТЕКСТА ==========
@bot.message_handler(func=lambda message: True)
def handle_text(message):
    """Обработка текстовых сообщений"""
    text = message.text.strip()
    user_id = message.from_user.id
    
    # Проверяем кнопки меню
    if text == "📸 Распознать фото":
        bot.send_message(message.chat.id,
                        "📸 **ОТПРАВЬТЕ ФОТО С ТЕКСТОМ:**\n\n"
                        "Сфотографируйте вывеску, меню, указатель или любой текст и отправьте сюда.",
                        parse_mode='Markdown')
        return
        
    elif text == "📝 Переводчик":
        bot.send_message(message.chat.id,
                        "📝 **РЕЖИМ ПЕРЕВОДЧИКА:**\n\n"
                        "Отправьте текст на любом языке для перевода.",
                        parse_mode='Markdown')
        return
        
    elif text == "🌍 Язык перевода":
        cmd_language(message)
        return
        
    elif text == "📚 История":
        cmd_history(message)
        return
        
    elif text == "❓ Помощь":
        cmd_help(message)
        return
    
    # Если текст короткий
    if len(text) < 2:
        bot.send_message(message.chat.id,
                        "❌ Текст слишком короткий.\n"
                        "Отправьте более длинное сообщение или фото с текстом.",
                        parse_mode='Markdown')
        return
    
    try:
        # Определяем язык
        detected = translator.detect(text)
        src_lang = detected.lang
        confidence = detected.confidence * 100
        
        # Получаем язык пользователя
        target_lang = get_user_language(user_id)
        
        # Переводим
        translation = translator.translate(text, src=src_lang, dest=target_lang)
        
        # Сохраняем в историю
        add_to_history(user_id, 'text', text, translation.text, src_lang, target_lang)
        
        # Формируем ответ
        lang_names = {
            'en': 'английский', 'ru': 'русский', 'de': 'немецкий',
            'fr': 'французский', 'es': 'испанский', 'ja': 'японский',
            'ko': 'корейский', 'it': 'итальянский'
        }
        
        src_name = lang_names.get(src_lang, src_lang)
        targ_name = lang_names.get(target_lang, target_lang)
        
        response = f"""
📝 **ИСХОДНЫЙ ТЕКСТ ({src_name.upper()}):**
`{text}`

🌍 **ОПРЕДЕЛЕН ЯЗЫК:** {src_name.upper()} (точность: {confidence:.1f}%)
🎯 **ПЕРЕВОД НА {targ_name.upper()}:**
{translation.text}
        """
        
        bot.reply_to(message, response, parse_mode='Markdown')
        
    except Exception as e:
        bot.reply_to(message, f"❌ **ОШИБКА ПЕРЕВОДА:**\n\n`{str(e)}`", parse_mode='Markdown')

# ========== CALLBACK ОБРАБОТЧИКИ ==========
@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    """Обработка inline-кнопок"""
    try:
        if call.data.startswith("lang_"):
            # Установка языка
            lang = call.data[5:]  # Убираем "lang_"
            user_id = call.from_user.id
            set_user_language(user_id, lang)
            
            lang_names = {
                'en': 'английский', 'de': 'немецкий', 'fr': 'французский',
                'es': 'испанский', 'ja': 'японский', 'ko': 'корейский',
                'ru': 'русский', 'it': 'итальянский', 'pt': 'португальский',
                'ar': 'арабский', 'tr': 'турецкий', 'zh-cn': 'китайский'
            }
            
            lang_name = lang_names.get(lang, lang)
            
            bot.answer_callback_query(call.id, f"✅ Язык: {lang_name}")
            bot.edit_message_text(
                f"🌍 **ЯЗЫК ПЕРЕВОДА УСТАНОВЛЕН:** {lang_name.upper()}",
                call.message.chat.id,
                call.message.message_id,
                parse_mode='Markdown'
            )
            
    except Exception as e:
        bot.answer_callback_query(call.id, f"❌ Ошибка: {str(e)[:50]}")

# ========== ЗАПУСК БОТА ==========
if __name__ == '__main__':
    print("=" * 60)
    print("🤖 LANGHELPER BOT - ПУТЕШЕСТВЕННИК С OCR")
    print("=" * 60)
    print(f"✅ Токен загружен")
    print(f"✅ Нейросеть EasyOCR инициализирована")
    print(f"✅ База данных: {DB_FILE}")
    print("\n⚠️  ИНФОРМАЦИЯ:")
    print("• EasyOCR работает на CPU (это нормально)")
    print("• GPU не требуется для проекта")
    print("• Распознавание работает медленнее на CPU")
    print("\n📱 ОСНОВНЫЕ ФУНКЦИИ:")
    print("• 📸 Распознавание текста с фото (OCR)")
    print("• 📝 Текстовый перевод")
    print("• 🌍 Автоопределение языка")
    print("• 📚 История переводов")
    print("\n🚀 Бот готов к работе!")
    print("Отправьте /start в Telegram для начала")
    print("\n🛑 Для остановки нажмите Ctrl+C")
    print("=" * 60)
    
    try:
        bot.infinity_polling()
    except KeyboardInterrupt:
        print("\n👋 Бот остановлен")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")