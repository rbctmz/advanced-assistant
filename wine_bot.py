import telebot
from dotenv import load_dotenv
import os
from yandex_cloud_ml_sdk import YCloudML
from pydantic import BaseModel, Field
import pandas as pd
from typing import Optional
import asyncio
from time import sleep
import signal
import logging
from threading import Lock

# Import search index components
from yandex_cloud_ml_sdk.search_indexes import (
    StaticIndexChunkingStrategy,
    HybridSearchIndexType,
    ReciprocalRankFusionIndexCombinationStrategy,
)

# Setup logging
logging.basicConfig(
    filename='wine_bot.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize environment and SDK
load_dotenv()

folder_id = os.getenv("folder_id")
api_key = os.getenv("api_key")
telegram_token = os.getenv("tg_token")

if not all([folder_id, api_key, telegram_token]):
    raise ValueError("Please set folder_id, api_key and tg_token in .env file")

sdk = YCloudML(folder_id=folder_id, auth=api_key)

model = sdk.models.completions("yandexgpt", model_version="rc")

# Add wine data processing
def process_wine_data(df):
    # Rename columns
    df.columns = ["Id", "Name", "Country", "Price", "WHPrice", "etc"]
    
    # Extract acidity
    acid_map = {
        "СХ": "Сухое", 
        "СЛ": "Сладкое", 
        "ПСХ": "Полусухое", 
        "ПСЛ": "Полусладкое"
    }
    df["Acidity"] = df["Name"].apply(
        lambda x: acid_map.get(x.split()[-1].replace("КР", ""), "")
    )
    
    # Extract color
    df["Color"] = df["Name"].apply(
        lambda x: (
            "Красное" if (x.split()[-1].startswith("КР") or x.split()[-2] == "КР") else ""
        )
    )
    logger.info(f"Wine data processed successfully. Found {len(df)} wines")
    return df

# Add country mapping
country_map = {
    "IT": "Италия", "FR": "Франция", "ES": "Испания",
    "RU": "Россия", "PT": "Португалия", "AR": "Армения",
    "CL": "Чили", "AU": "Австрия", "GE": "Грузия",
    "ZA": "ЮАР", "US": "США", "NZ": "Новая Зеландия",
    "DE": "Германия", "AT": "Австрия", "IL": "Израиль",
    "BG": "Болгария", "GR": "Греция", "AU": "Австралия",
}

revmap = {v.lower(): k for k, v in country_map.items()}

# Add wine search function

# Define the function to find wines based on user input
def find_wines(req):
    x = wine_df.copy()
    if req.country:
        country_query = req.country.lower().strip()
        if country_query in revmap:
            print(f"Searching for {req.country}")
            x = x[x["Country"] == revmap[country_query]]
            print(f"Found {len(x)} wines for {req.country}")
    if req.acidity:
        print(f"Searching for {req.acidity}")
        x = x[x["Acidity"] == req.acidity.capitalize()]
        print(f"Found {len(x)} wines for {req.acidity}")
    if req.color:
        print(f"Searching for {req.color}")
        x = x[x["Color"] == req.color.capitalize()]
        print(f"Found {len(x)} wines for {req.color}")
    if req.name:
        print(f"Searching for {req.name}")
        x = x[x["Name"].apply(lambda x: req.name.lower() in x.lower())]
        print(f"Found {len(x)} wines for {req.name}")
    if req.sort_order and len(x)>0:
        if req.sort_order == "cheapest":
            x = x.sort_values(by="Price")
        elif req.sort_order == "most expensive":
            x = x.sort_values(by="Price", ascending=False)
    if x is None or len(x) == 0:
        return "Подходящих вин не найдено"
    return "Найдено всего наименований: " + str(len(x)) + ". Вот какие вина были найдены:\n" + "\n".join(
        [
            f"{z['Name']} ({country_map.get(z['Country'],'Неизвестно')}) - {z['Price']}"
            for _, z in x.head(10).iterrows()
        ]
    )

# 1. First, load and process all files
from glob import glob # Import glob for file searching
from tqdm.auto import tqdm # Import tqdm for progress bar

def get_token_count(filename):
    with open(filename, "r", encoding="utf8") as f:
        return len(model.tokenize(f.read()))

def get_file_len(filename):
    with open(filename, encoding="utf-8") as f:
        return len(f.read())

print("Loading files...")

d = [
    {
        "File": fn,
        "Tokens": get_token_count(fn),
        "Chars": get_file_len(fn),
        "Category": os.path.basename(os.path.dirname(fn)),
    }
    for fn in glob("data\\*\\*.md")
]

df = pd.DataFrame(d)
print(f"Loaded {len(df)} files")

# 2. Upload files and create search index
print("Uploading files to cloud...")

def upload_file(filename):
    print(f"Uploading {filename}...")
    return sdk.files.upload(filename, ttl_days=1, expiration_policy="static")

df["Uploaded"] = df["File"].apply(upload_file)

print("Creating search index...")
op = sdk.search_indexes.create_deferred(
    df[df["Category"] == "wines"]["Uploaded"],
    index_type=HybridSearchIndexType(
        chunking_strategy=StaticIndexChunkingStrategy(
            max_chunk_size_tokens=1000, 
            chunk_overlap_tokens=100
        ),
        combination_strategy=ReciprocalRankFusionIndexCombinationStrategy(),
    ),
)
index = op.wait()

# 3. Add region data
print("Adding region data...")
op = index.add_files_deferred(df[df["Category"]=="regions"]["Uploaded"])
xfiles = op.wait()

# 4. Process and add food-wine pairing data
print("Processing food-wine pairing data...")
with open("data/food_wine_table.md", encoding="utf-8") as f:
    food_wine = f.readlines()
header = food_wine[:2]

chunk_size = 600 * 3
s = header.copy()
uploaded_foodwine = []
for x in food_wine[2:]:
    s.append(x)
    if len("".join(s)) > chunk_size:
        id = sdk.files.upload_bytes(
            "".join(s).encode(), 
            ttl_days=5, 
            expiration_policy="static",
            mime_type="text/markdown",
        )
        uploaded_foodwine.append(id)
        s = header.copy()

print("Adding food-wine pairing data...")
op = index.add_files_deferred(uploaded_foodwine)
xfiles = op.wait()

print(f"RAG initialization complete with {len(df)} files")
logger.info("RAG initialization complete with %d files", len(df))

# 5. Load wine data
def load_wine_data():
    try:
        df = pd.read_excel("data/wine-price.xlsx")
        return process_wine_data(df)
    except Exception as e:
        print(f"Error loading wine data: {e}")
        return pd.DataFrame()

wine_df = load_wine_data()
if wine_df.empty:
    raise ValueError("Failed to load wine data. Please check data/wine-price.xlsx file")
print(f"Loaded {len(wine_df)} wines")
logger.info("Loaded %d wines", len(wine_df))

# 6. Create agent

class Agent:
    def __init__(self, assistant=None, instruction=None, search_index=None, tools=None):

        self.thread = None

        if assistant:
            self.assistant = assistant
        else:
            if tools:
                self.tools = {x.__name__: x for x in tools}
                tools = [sdk.tools.function(x) for x in tools]
            else:
                self.tools = {}
                tools = []
            if search_index:
                tools.append(sdk.tools.search_index(search_index))
            self.assistant = create_assistant(model, tools)

        if instruction:
            self.assistant.update(instruction=instruction)

    def get_thread(self, thread=None):
        if thread is not None:
            return thread
        if self.thread == None:
            self.thread = create_thread()
        return self.thread

    def __call__(self, message: str, thread=None):
        thread = self.get_thread(thread)
        thread.write(message)
        run = self.assistant.run(thread)
        res = run.wait()
        if res.tool_calls:
            result = []
            for f in res.tool_calls:
                print(
                    f" + Вызываем функцию {f.function.name}, args={f.function.arguments}"
                )
                fn = self.tools[f.function.name]
                obj = fn(**f.function.arguments)
                x = obj.process(thread)
                result.append({"name": f.function.name, "content": x})
            run.submit_tool_results(result)
            #time.sleep(3)
            res = run.wait()
        return res.text

    def restart(self):
        if self.thread:
            self.thread.delete()
            self.thread = sdk.threads.create(
                name="Test", ttl_days=1, expiration_policy="static"
            )

    def done(self, delete_assistant=False):
        if self.thread:
            self.thread.delete()
        if delete_assistant:
            self.assistant.delete()

handover = False

class Handover(BaseModel):
    """Эта функция позволяет передать диалог человеку-оператору поддержки"""

    reason: str = Field(
        description="Причина для вызова оператора", default="не указана"
    )

    def process(self, thread):
        global handover
        handover = True
        return f"Я побежала вызывать оператора, ваш {thread.id=}, причина: {self.reason}"

# Add cart functionality
carts = {}

class AddToCart(BaseModel):
    """Эта функция позволяет положить или добавить вино в корзину"""
    wine_name: str = Field(description="Точное название вина", default=None)
    count: int = Field(description="Количество бутылок", default=1)
    price: float = Field(description="Цена вина", default=None)

    def process(self, thread):
        if thread.id not in carts:
            carts[thread.id] = []
        carts[thread.id].append(self)
        logger.info(f"Wine {self.wine_name} added to cart for thread {thread.id}, count: {self.count}, price: {self.price}")
        print(f"Wine {self.wine_name} added to cart for thread {thread.id}, count: {self.count}, price: {self.price}")
        return f"Вино {self.wine_name} добавлено в корзину, число бутылок: {self.count} по цене {self.price}"

class ShowCart(BaseModel):
    """Эта функция позволяет показать содержимое корзины"""
    def process(self, thread):
        if thread.id not in carts or len(carts[thread.id]) == 0:
            return "Корзина пуста"
        return "В корзине находятся следующие вина:\n" + "\n".join(
            [f"{x.wine_name}, число бутылок: {x.count}" for x in carts[thread.id]]
        )

class ClearCart(BaseModel):
    """Эта функция очищает корзину"""
    def process(self, thread):
        if thread.id in carts:
            carts[thread.id].clear()
        return "Корзина очищена"

# Initialize bot
bot = telebot.TeleBot(telegram_token)

def create_assistant(model, tools=None):
    kwargs = {}
    if tools and len(tools) > 0:
        kwargs = {"tools": tools}
    return sdk.assistants.create(
        model, ttl_days=1, expiration_policy="since_last_active", **kwargs
    )

# Initialize threads dictionary for user conversations
threads = {}

def create_thread():
    return sdk.threads.create(ttl_days=1, expiration_policy="static")

def get_thread(chat_id):
    if chat_id in threads.keys():
        return threads[chat_id]
    t = create_thread()
    # Write wine data context to thread
    if not wine_df.empty:
        t.write(f"Available wines:\n{wine_df.to_string()}")
    print(f"New thread {t.id=} created")
    threads[chat_id] = t
    return t

def write_message_to_thread(thread, message, max_retries=3):
    """Write user message to thread with retry logic"""
    for attempt in range(max_retries):
        try:
            thread.write(message.text)
            return True
        except Exception as e:
            if "is locked by run" in str(e):
                print(f"Thread locked, attempt {attempt + 1}/{max_retries}")
                sleep(1)  # Wait before retry
                continue
            print(f"Error writing to thread: {e}")
            return False
    return False

def process_assistant_response(run, default_msg="I apologize, but I'm having trouble processing your request."):
    """Process assistant run result with error handling"""
    try:
        result = run.wait()
        if not result or not result.text:
            return default_msg
        return result.text
    except Exception as e:
        print(f"Error processing assistant response: {e}")
        return default_msg

# Define search tool
class SearchWinePriceList(BaseModel):
    """Эта функция позволяет искать вина в прайс-листе по одному или нескольким параметрам."""
    name: str = Field(description="Название вина", default=None)
    country: str = Field(description="Страна", default=None) 
    acidity: str = Field(description="Кислотность (сухое, полусухое, сладкое, полусладкое)", default=None)
    color: str = Field(description="Цвет вина (красное, белое, розовое)", default=None)
    sort_order: str = Field(description="Порядок сортировки (самое дорогое, самое дешевое) | Sort order (most expensive, cheapest)", default=None)
    what_to_return: str = Field(description="Что вернуть (информация о вине или цена)", default=None)

    def process(self, thread):
        """Process the search request"""
        return find_wines(self)

# Create agent with instructions

instruction = """
Ты - опытный сомелье, продающий вино в магазине. Твоя задача - отвечать на вопросы пользователя
про вина, рекомендовать лучшие вина к еде, а также искать вина в прайс-листе нашего магазина,
а также проактивно предлагать пользователю приобрести вина, отвечающие его потребностям. В ответ
на сообщение /start поинтересуйся, что нужно пользователю, предложи ему какой-то
интересный вариант сочетания еды и вине, и попытайся продать ему вино.
Посмотри на всю имеющуюся в твоем распоряжении информацию
и выдай одну или несколько лучших рекомендаций.
Если вопрос касается конкретных вин или цены, то вызови функцию SearchWinePriceList.
Для передачи управления оператору - вызови фукцию Handover. Для добавления вина в корзину
используй AddToCart. Для просмотра корзины: ShowCart.
Если что-то непонятно, то лучше уточни информацию у пользователя.
"""
wine_agent = Agent(
    instruction=instruction,
    search_index=index,
    tools=[SearchWinePriceList, 
           Handover, 
           AddToCart, 
           ShowCart,
           ClearCart],
)

# Add assistants cache at the top level
assistants = {}
shared_assistant = None

def get_assistant(chat_id):
    """Get or create assistant for chat"""
    global shared_assistant
    if shared_assistant is None:
        shared_assistant = wine_agent()
    return shared_assistant

# Add thread locks dictionary
thread_locks = {}

def get_or_create_thread(chat_id):
    """Get existing thread or create new one with lock management"""
    global threads, thread_locks
    
    # Create new lock if needed
    if chat_id not in thread_locks:
        thread_locks[chat_id] = Lock()
    
    with thread_locks[chat_id]:
        # Check if thread exists and is not locked
        if chat_id in threads:
            try:
                # Test if thread is usable
                threads[chat_id].write("test")
                threads[chat_id].delete_messages()
                return threads[chat_id]
            except Exception:
                # Thread is locked or invalid, create new one
                logger.info(f"Creating new thread for chat {chat_id}")
                threads[chat_id] = sdk.threads.create(ttl_days=1, expiration_policy="static")
        else:
            # Create new thread
            threads[chat_id] = sdk.threads.create(ttl_days=1, expiration_policy="static")
            logger.info(f"New thread created: {threads[chat_id].id}")
        
        return threads[chat_id]

# Bot command handlers
@bot.message_handler(commands=["start"])
def start(message):
    try:
        t = get_thread(message.chat.id)
        assistant = get_assistant(message.chat.id)
        print(f"Starting on thread {t.id=}, msg={message.text}")
        
        t.write("Новый пользователь начал разговор")
        run = assistant.run(t)
        result = run.wait(timeout=30)
        
        response = result.text if result and result.text else "Здравствуйте! Я - ваш персональный сомелье. Как я могу помочь вам выбрать вино?"
        bot.send_message(message.chat.id, response)
        
    except Exception as e:
        print(f"Error in start handler: {e}")
        bot.send_message(
            message.chat.id,
            "Здравствуйте! К сожалению, возникли технические проблемы. Попробуйте начать разговор через минуту."
        )

@bot.message_handler(commands=["stop"])
def stop(message):
    t = get_thread(message.chat.id)
    t.delete()
    del threads[message.chat.id]
    bot.send_message(message.chat.id, "Ваш разговор завершен. Если у вас есть другие вопросы, не стесняйтесь обращаться!")
    del assistants[message.chat.id]
    print(f"Thread {t.id} deleted and assistant removed for chat {message.chat.id}")
    logger.info(f"Thread {t.id} deleted and assistant removed for chat {message.chat.id}")

def wait_for_thread_unlock(thread, max_wait=10):
    """Wait for thread to unlock with timeout"""
    for i in range(max_wait):
        try:
            thread.write("Checking thread status")
            thread.delete_messages()  # Clean up status check message
            return True
        except Exception as e:
            if "is locked by run" in str(e):
                print(f"Thread still locked, waiting... {i+1}/{max_wait}")
                sleep(1)
                continue
            return False
    return False

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        # Get thread with lock management
        t = get_or_create_thread(message.chat.id)
        print(f"Handling message on thread {t.id=}, msg={message.text}")
        logger.info(f"Processing message on thread {t.id=}, msg={message.text}")
        
        answer = wine_agent(message.text, thread=t)
        print(f"Answer from agent: {answer}")
        logger.info(f"Answer from agent: {answer}")
        bot.send_message(message.chat.id, answer)
     
    except Exception as e:
        logger.error(f"Error in message handler: {e}")
        bot.send_message(
            message.chat.id,
            "Произошла ошибка. Пожалуйста, попробуйте через минуту."
        )

def signal_handler(signum, frame):
    print("\nShutting down bot gracefully...")
    # Cleanup threads and assistants
    threads.clear()
    if 'wine_assistant' in globals():
        print("Cleaning up assistant...")
    print("Bot shutdown complete")
    exit(0)

# Update main block
if __name__ == "__main__":
    try:
        # Register signal handler
        signal.signal(signal.SIGINT, signal_handler)
        print("Initializing wine assistant...")
        print("Bot is running... Press Ctrl+C to exit gracefully")
        bot.polling(none_stop=True)
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        print(f"Failed to start bot: {e}")