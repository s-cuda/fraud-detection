import redis
import json
import os

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT  = int(os.getenv('REDIS_PORT', 6379))

# TTL — how long card stats live in Redis before expiring
# 7 days — if a card hasn't transacted in 7 days, its stats reset
CARD_TTL = 60 * 60 * 24 * 7

class FeatureStore:

    def __init__(self):
        self.client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True
        )
        # Test connection
        self.client.ping()
        print(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")

    def update_card_stats(self, card_id: str, amount: float, 
                          timestamp: float, email_domain: str,
                          addr: str):
        """
        After each transaction, update rolling stats for this card.
        Called by the API every time a transaction is scored.
        """
        key = f"card:{card_id}"

        # Get existing stats
        existing = self.client.get(key)
        if existing:
            stats = json.loads(existing)
        else:
            stats = {
                'txn_count':     0,
                'amt_sum':       0.0,
                'amt_sq_sum':    0.0,
                'first_seen':    timestamp,
                'emails':        [],
                'addrs':         []
            }

        # Update rolling stats
        stats['txn_count']  += 1
        stats['amt_sum']    += amount
        stats['amt_sq_sum'] += amount ** 2
        stats['last_seen']   = timestamp

        # Track unique emails and addresses (keep last 20)
        if email_domain and email_domain not in stats['emails']:
            stats['emails'].append(email_domain)
            stats['emails'] = stats['emails'][-20:]

        if addr and str(addr) not in stats['addrs']:
            stats['addrs'].append(str(addr))
            stats['addrs'] = stats['addrs'][-20:]

        # Save back to Redis with TTL
        self.client.setex(key, CARD_TTL, json.dumps(stats))

    def get_card_features(self, card_id: str, 
                          current_amount: float,
                          current_timestamp: float) -> dict:
        """
        Retrieve behavioral features for this card.
        Returns computed features ready to feed into the model.
        """
        key     = f"card:{card_id}"
        existing = self.client.get(key)

        if not existing:
            # Card never seen before — return default features
            return {
                'card_txn_count':       1,
                'amt_to_card_mean_ratio': 1.0,
                'amt_z_score_card':      0.0,
                'card_time_since_first': 0.0,
                'card_unique_email':     1,
                'card_unique_addr':      1,
                'card_amt_std':          0.0,
                'is_new_card':           1,
            }

        stats    = json.loads(existing)
        count    = stats['txn_count']
        amt_mean = stats['amt_sum'] / count if count > 0 else current_amount

        # Standard deviation from sum of squares
        if count > 1:
            variance = (stats['amt_sq_sum'] / count) - (amt_mean ** 2)
            amt_std  = max(variance, 0) ** 0.5
        else:
            amt_std  = 0.0

        time_since_first = current_timestamp - stats.get('first_seen', current_timestamp)

        return {
            'card_txn_count':         count,
            'amt_to_card_mean_ratio': current_amount / (amt_mean + 1),
            'amt_z_score_card':       (current_amount - amt_mean) / (amt_std + 1),
            'card_time_since_first':  time_since_first,
            'card_unique_email':      len(stats.get('emails', [])),
            'card_unique_addr':       len(stats.get('addrs',  [])),
            'card_amt_std':           amt_std,
            'is_new_card':            1 if time_since_first < 30 * 86400 else 0,
        }

    def get_stats(self) -> dict:
        """How many cards are currently tracked in Redis."""
        keys = self.client.keys('card:*')
        return {
            'cards_tracked': len(keys),
            'redis_host':    REDIS_HOST,
            'redis_port':    REDIS_PORT,
        }