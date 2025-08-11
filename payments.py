import os
from typing import Dict, Optional
from dotenv import load_dotenv
import stripe

load_dotenv()

STRIPE_API_KEY = os.getenv("STRIPE_API_KEY")


def create_payment_link(
    amount_cents: int,
    currency: str,
    product_name: str,
    quantity: int = 1,
    price_id: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    """Create a Stripe Payment Link and return its URL.

    If `price_id` is provided, the link will use that existing Price. Otherwise, this
    creates a Payment Link using inline price data (no need to create products/prices first).

    Args:
        amount_cents: Price amount in the smallest currency unit (e.g., cents for USD).
        currency: Three-letter ISO currency code (e.g., "usd").
        product_name: Name to display for the line item.
        quantity: Quantity for the line item (default 1).
        price_id: Optional existing Stripe Price ID to use instead of inline price data.
        metadata: Optional key-value pairs to attach to the Payment Link.

    Returns:
        The URL string for the created Payment Link.

    Raises:
        ValueError: If inputs are invalid or API key is missing.
        Exception: Propagates Stripe API errors with context.
    """
    if amount_cents <= 0 and price_id is None:
        raise ValueError("amount_cents must be > 0 when price_id is not provided")
    if quantity <= 0:
        raise ValueError("quantity must be > 0")
    if not currency:
        raise ValueError("currency is required")
    if not product_name and price_id is None:
        raise ValueError("product_name is required when price_id is not provided")

    stripe.api_key = STRIPE_API_KEY

    try:
        if price_id:
            payment_link = stripe.PaymentLink.create(
                line_items=[{"price": price_id, "quantity": quantity}],
                metadata=metadata or {},
            )
        else:
            payment_link = stripe.PaymentLink.create(
                line_items=[
                    {
                        "price_data": {
                            "currency": currency.lower(),
                            "product_data": {"name": product_name},
                            "unit_amount": amount_cents,
                        },
                        "quantity": quantity,
                    }
                ],
                metadata=metadata or {},
            )

        url: str = payment_link["url"]
        if not url:
            raise RuntimeError("Stripe returned an empty URL for the Payment Link")
        return url

    except Exception as exc:
        raise RuntimeError(f"Failed to create Stripe Payment Link: {exc}") from exc


