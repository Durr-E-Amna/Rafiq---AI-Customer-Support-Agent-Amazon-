# Order tracking

Every order gets a tracking number once it's handed to a carrier — not at the moment the order is placed. Before that, the order just shows as "processing," since there's nothing to track yet.

## What the order statuses mean

- **Processing** — order confirmed, being prepared for shipment
- **Shipped** — handed to the carrier, tracking number is now active
- **In transit** — moving between carrier hubs
- **Out for delivery** — with the local courier for final delivery, usually same-day
- **Delivered** — confirmed received
- **Delivery attempt failed** — courier couldn't complete delivery (no one available, access issue, refused delivery)

## Delivery timelines and guarantees

Standard delivery windows vary by item, shipping speed selected, and destination — there's no single fixed number, since some items ship from different fulfillment centers than others. Where a specific delivery date was guaranteed at checkout (shown clearly at the time, with an associated cost), and a delivery attempt isn't made by that date, the shipping fee for that order is refunded automatically — that's a guarantee, not a goodwill gesture, so it shouldn't be treated as something the customer needs to negotiate for.

International orders (shipped from a different country than the delivery address) generally take noticeably longer than domestic ones, since they go through customs clearance in addition to standard transit — often an extra week or more on top of the domestic estimate. The exact delivery window for an international item is shown on the product page and order confirmation before checkout, and that estimate already accounts for the customs step, so it should be treated as the real expectation rather than a worst case.

## If tracking isn't updating

Tracking systems sync periodically, not instantly. If a status hasn't changed in the first day right after shipping, that's usually just a sync delay, not a lost package. If there's been no movement for 3+ business days while marked "in transit," that's worth escalating for a manual check with the carrier rather than continuing to reassure the customer it's probably fine.

## Failed delivery attempts

After repeated failed delivery attempts, the package is typically returned to the fulfillment center and the order is marked undelivered. No refund is issued automatically in this case — the customer needs to request redelivery or initiate a standard return/refund process once the package is confirmed returned to the warehouse.
