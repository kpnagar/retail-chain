# Create the in memory store for orders, carts etc

orders = [{
    "id": "e3d5ff96-7aa5-4a92-af8c-3a4059a022b9",
    "product": {
        "id": 101,
        "title": "Apple iPhone 13 Pro Max",
        "price": 1099.00,
        "description": "6.7-inch Super Retina XDR display, A15 Bionic chip, Pro camera system with 12MP camera.",
        "images": ["iphone_13_pro_max_front.jpg", "iphone_13_pro_max_back.jpg"],
        "creationAt": "2024-03-28T10:00:00",
        "updatedAt": "2024-03-28T10:30:00",
        "category": {
            "id": 1,
            "name": "Electronics",
            "image": "electronics_category.jpg",
            "creationAt": "2024-03-28T09:00:00",
            "updatedAt": "2024-03-28T09:30:00"
        }
    },
    "total_amount": 1099.00,
    "order_date": "2024-03-28T11:00:00",
    "status": "processing"
},
    {
        "id": "5d7e901b-f17e-4f15-b9b8-7415a51cde1b",
        "product": {
            "id": 202,
            "title": "Nike Air Max 270",
            "price": 150.00,
            "description": "Men's Running Shoes, lightweight, breathable, Max Air unit provides comfortable cushioning.",
            "images": ["nike_air_max_270_black.jpg", "nike_air_max_270_white.jpg"],
            "creationAt": "2024-03-28T11:00:00",
            "updatedAt": "2024-03-28T11:30:00",
            "category": {
                "id": 2,
                "name": "Footwear",
                "image": "footwear_category.jpg",
                "creationAt": "2024-03-28T10:00:00",
                "updatedAt": "2024-03-28T10:30:00"
            }
        },
        "total_amount": 150.00,
        "order_date": "2024-03-28T12:00:00",
        "status": "shipped"
    },
    {
        "id": "8ff5276d-2ef8-4522-a92c-f9a9e2063a6c",
        "product": {
            "id": 303,
            "title": "Fender American Professional II Stratocaster",
            "price": 1499.99,
            "description": "Electric Guitar, Alder body, Deep 'C' neck profile, V-Mod II single-coil pickups.",
            "images": ["fender_stratocaster_sunburst.jpg", "fender_stratocaster_black.jpg"],
            "creationAt": "2024-03-28T12:00:00",
            "updatedAt": "2024-03-28T12:30:00",
            "category": {
                "id": 3,
                "name": "Musical Instruments",
                "image": "musical_instruments_category.jpg",
                "creationAt": "2024-03-28T11:00:00",
                "updatedAt": "2024-03-28T11:30:00"
            }
        },
        "total_amount": 1499.99,
        "order_date": "2024-03-28T13:00:00",
        "status": "delivered"
    },
    {
        "id": "12345678-abcd-5678-efgh-abcdefgh1234",
        "product": {
            "id": 404,
            "title": "Sony Alpha A7 III Mirrorless Camera",
            "price": 1999.99,
            "description": "24MP full-frame sensor, 4K video capability, 693-point autofocus system.",
            "images": ["sony_a7iii_front.jpg", "sony_a7iii_back.jpg"],
            "creationAt": "2024-03-28T13:00:00",
            "updatedAt": "2024-03-28T13:30:00",
            "category": {
                "id": 1,
                "name": "Electronics",
                "image": "electronics_category.jpg",
                "creationAt": "2024-03-28T09:00:00",
                "updatedAt": "2024-03-28T09:30:00"
            }
        },
        "total_amount": 1999.99,
        "order_date": "2024-03-28T14:00:00",
        "status": "cancelled"
    },
    {
        "id": "98765432-dcba-8765-hgfe-dcbahgfedcba",
        "product": {
            "id": 505,
            "title": "Cuisinart TOA-60 Convection Toaster Oven Airfryer",
            "price": 199.95,
            "description": "7 Functions in 1 - Airfry, Convection Bake, Convection Broil, Bake, Broil, Warm, and Toast.",
            "images": ["cuisinart_toaster_oven_airfryer_front.jpg", "cuisinart_toaster_oven_airfryer_side.jpg"],
            "creationAt": "2024-03-28T14:00:00",
            "updatedAt": "2024-03-28T14:30:00",
            "category": {
                "id": 4,
                "name": "Home & Kitchen",
                "image": "home_kitchen_category.jpg",
                "creationAt": "2024-03-28T13:00:00",
                "updatedAt": "2024-03-28T13:30:00"
            }
        },
        "total_amount": 199.95,
        "order_date": "2024-03-28T15:00:00",
        "status": "pending"
    }
]
