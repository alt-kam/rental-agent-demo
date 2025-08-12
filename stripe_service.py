import os
from typing import Dict, Any
import stripe

class StripeService:
    def __init__(self):
        # Read environment variables. Do not hardcode secrets in code.
        stripe.api_key = os.getenv('STRIPE_SECRET_KEY', '')
        self.publishable_key = os.getenv('STRIPE_PUBLISHABLE_KEY', '')
    
    def create_checkout_session(self, booking_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a Stripe Checkout Session for LivedIn booking payment
        """
        try:
            # Ensure latest API key is set from env for this call
            stripe.api_key = os.getenv('STRIPE_SECRET_KEY', '')
            # Create checkout session
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': f"🏠 {booking_details['property_name']}",
                            'description': f"Stay from {booking_details['check_in']} to {booking_details['check_out']}",
                            'images': booking_details.get('property_images', []),
                        },
                        'unit_amount': int(booking_details['amount'] * 100),  # Convert to cents
                    },
                    'quantity': 1,
                }],
                mode='payment',
                success_url=booking_details.get('success_url', 'https://your-domain.com/success?session_id={CHECKOUT_SESSION_ID}'),
                cancel_url=booking_details.get('cancel_url', 'https://your-domain.com/cancel'),
                customer_email=booking_details.get('guest_email'),
                metadata={
                    'property_id': booking_details.get('property_id', ''),
                    'guest_name': booking_details.get('guest_name', ''),
                    'check_in': booking_details['check_in'],
                    'check_out': booking_details['check_out'],
                    'original_price': str(booking_details.get('original_price', '')),
                    'negotiated_price': str(booking_details['amount']),
                    'booking_type': 'livedin_ai_booking'
                },
                payment_intent_data={
                    'metadata': {
                        'integration_check': 'livedin_demo',
                    }
                }
            )
            
            return {
                'success': True,
                'session_id': session.id,
                'session_url': session.url,
                'payment_intent_id': session.payment_intent
            }
            
        except stripe.error.StripeError as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def retrieve_session(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieve checkout session details
        """
        try:
            session = stripe.checkout.Session.retrieve(session_id)
            return {
                'success': True,
                'session': {
                    'id': session.id,
                    'payment_status': session.payment_status,
                    'amount_total': session.amount_total,
                    'currency': session.currency,
                    'customer_email': session.customer_details.email if session.customer_details else None,
                    'metadata': session.metadata,
                    'payment_intent_id': session.payment_intent
                }
            }
        except stripe.error.StripeError as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def simulate_test_scenarios(self, scenario: str = 'success') -> Dict[str, str]:
        """
        Return test card numbers for different scenarios
        """
        test_cards = {
            'success': {
                'number': '4242424242424242',
                'description': 'Payment succeeds',
                'instructions': 'Use any future expiry date and any 3-digit CVC'
            },
            'decline': {
                'number': '4000000000000002',
                'description': 'Payment is declined',
                'instructions': 'Use any future expiry date and any 3-digit CVC'
            },
            'insufficient_funds': {
                'number': '4000000000009995',
                'description': 'Insufficient funds',
                'instructions': 'Use any future expiry date and any 3-digit CVC'
            },
            'expired': {
                'number': '4000000000000069',
                'description': 'Expired card',
                'instructions': 'Use any past expiry date and any 3-digit CVC'
            },
            '3d_secure': {
                'number': '4000000000003220',
                'description': 'Requires 3D Secure authentication',
                'instructions': 'Use any future expiry date and any 3-digit CVC'
            }
        }
        
        return test_cards.get(scenario, test_cards['success'])
    
    def create_payment_link(self, booking_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Alternative: Create a Payment Link (simpler approach)
        """
        try:
            # Create product first
            product = stripe.Product.create(
                name=f"🏠 {booking_details['property_name']}",
                description=f"Stay from {booking_details['check_in']} to {booking_details['check_out']}",
                metadata={
                    'property_id': booking_details.get('property_id', ''),
                    'guest_name': booking_details.get('guest_name', ''),
                }
            )
            
            # Create price
            price = stripe.Price.create(
                unit_amount=int(booking_details['amount'] * 100),
                currency='usd',
                product=product.id,
            )
            
            # Create payment link
            payment_link = stripe.PaymentLink.create(
                line_items=[{
                    'price': price.id,
                    'quantity': 1,
                }],
                metadata={
                    'property_id': booking_details.get('property_id', ''),
                    'guest_name': booking_details.get('guest_name', ''),
                    'check_in': booking_details['check_in'],
                    'check_out': booking_details['check_out'],
                    'booking_type': 'livedin_ai_booking'
                }
            )
            
            return {
                'success': True,
                'payment_link_url': payment_link.url,
                'payment_link_id': payment_link.id
            }
            
        except stripe.error.StripeError as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }

# Initialize the service
stripe_service = StripeService()
