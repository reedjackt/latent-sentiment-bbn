select
    account_id,
    domain,
    page_views,
    email_opens
from {{ source('raw', 'accounts') }}
