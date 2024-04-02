from typing import List

import requests
from schemas import Product, Order
from store import orders
from datetime import datetime
from scheduler import Scheduler
from sentence_transformers import SentenceTransformer, util
from langchain_community.llms.ollama import Ollama

llm = Ollama(model="gemma:2b")
schedule = Scheduler()


def fetch_products_from_api(item: str):
    url = "https://api.escuelajs.co/api/v1/products/"
    params = {"title": item}

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        products = [Product(**item) for item in data]
        if not products:
            return None
        print(f"The following items are available for purchase. \n{data}")
        return products
    else:
        print("Error:", response.status_code)


def get_product_of_choice(products: List[Product]):
    choice = int(input("Type the Product ID of the product of your choice: "))
    for product in products:
        if product.id == choice:
            return product


def place_scheduled_order(product, placed_at):
    print(place_order(product, placed_at))


def place_order(product, placed_at):
    order = Order(product=product, total_amount=product.price, order_date=placed_at,
                  status="processing")
    orders.append(order.model_dump())
    return f"Your order for {product.title} has been placed. OrderID for your purchase is: {order.id}"


def chitchat(user_query: str) -> str:
    """
    Do simple chitchat with user respectfully and honestly.

    :param user_query: The query provided by the user.
    :type user_query: str
    :return: Respectful response of user's query in string format
    """
    return llm(user_query)


def product_purchase(item: str) -> str:
    """
    Use this when user wants to purchase a product.
    :param item: Type of item user wants to buy eg. phone, table etc
    :return order_id: ID of the order placed by the user
    """
    products = fetch_products_from_api(item)
    if not products:
        return "Unable to find the product you're looking for..."
    product = get_product_of_choice(products)
    return place_order(product, datetime.utcnow())


def schedule_purchase(item: str, datetime_to_schedule: str) -> bool:
    """
    Schedule a purchase of an item at a given datetime.

    Args:
    - item (str): The name or identifier of the item to be purchased.
    - datetime_to_schedule (str): The date when order is to be scheduled in "YYYY-MM-DD" format.

    Returns:
    - bool: True if the purchase was successfully scheduled, False otherwise.
    """
    products = fetch_products_from_api(item)
    if not products:
        return False
    product = get_product_of_choice(products)
    datetime_object = datetime.strptime(datetime_to_schedule, "%Y-%m-%d")
    order_purchased = schedule.once(datetime_object, place_scheduled_order,
                                    kwargs={"product": product, "placed_at": datetime_to_schedule})
    if not order_purchased:
        return False
    return True


def price_tracking(item: str) -> str:
    """
    Track a product price and notify user for price drop

    Args:
    - item (str): The name or identifier of the item to be purchased.

    Returns:
    - bool: True if the purchase was successfully added to card, False otherwise.
    """
    products = fetch_products_from_api(item)
    if not products:
        return "Unable to find the product you're looking for..."
    product = get_product_of_choice(products)
    return f"Your product {product.title} added in cart for price tracking"


def order_tracking(user_query: str, product_name: str):
    """
    use when user wants to retrieve order details

    :param user_query: user's original query
    :param product_name: the product whose status the user wants to retrieve
    :return: status: status of the order
    """
    product_list = [{"id": order['id'], "product_name": order["product"]["title"], "status": order["status"]} for order
                    in orders]
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    product_names = [order["product_name"] for order in product_list]
    embeddings = model.encode(product_names, convert_to_tensor=True)
    input_embedding = model.encode(product_name, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(input_embedding, embeddings)
    highest_similarity_index = cosine_scores.argmax().item()
    order_details = product_list[highest_similarity_index]
    prompt = f"""Answer the following question based on the given context. Answer in the tone of an helpful assistant.Keep the responses very short in plainstring format, only tell user about the order.
    ### Context:
    {order_details}
    
    ### UserQuery:
    {user_query}
    
    ### Response: 
    """
    print(order_details)
    return llm(prompt)
