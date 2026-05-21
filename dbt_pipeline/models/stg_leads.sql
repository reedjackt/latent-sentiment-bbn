with source as (
    select * from {{ source('raw', 'leads') }}
),

normalized as (
    select
        *,
        trim(cast(lead_id as varchar)) as lead_id_text,
        nullif(
            regexp_extract(trim(cast(lead_id as varchar)), '(?i)^LD-?(\d{1,8})$', 1),
            ''
        ) as lead_id_digits,
        lower(trim(cast(marketing_channel as varchar))) as marketing_channel_text,
        lower(trim(cast(campaign_tier as varchar))) as campaign_tier_text,
        upper(trim(cast(region as varchar))) as region_text,
        lower(trim(cast(replied_within_7d as varchar))) as reply_text,
        lower(
            regexp_replace(
                regexp_replace(
                    regexp_replace(trim(cast(job_title_raw as varchar)), '_', ' ', 'g'),
                    '\s*/\s*acting$',
                    ''
                ),
                '\s*\(interim\)$',
                ''
            )
        ) as job_title_text,
        lower(
            regexp_replace(
                regexp_replace(
                    regexp_replace(trim(cast(employee_band_messy as varchar)), '\s+to\s+', '-', 'g'),
                    '\s+',
                    '',
                    'g'
                ),
                '^see linkedin$',
                'seelinkedin'
            )
        ) as employee_band_text
    from source
)

select
    case
        when lead_id_digits is null then null
        else 'LD-' || lpad(lead_id_digits, 8, '0')
    end as lead_id_clean,
    regexp_matches(lead_id_text, '^LD-\d{8}$') as lead_id_was_valid,
    lead_id_digits is null as lead_id_malformed,
    captured_at,
    captured_at is null as captured_at_missing,
    cast(captured_at as date) as captured_date,
    extract(year from captured_at) as captured_year,
    extract(month from captured_at) as captured_month,
    lower(
        regexp_replace(
            regexp_replace(trim(cast(company_name_messy as varchar)), '™', '', 'g'),
            '\.$',
            ''
        )
    ) as company_name_clean,
    regexp_replace(lower(trim(cast(company_domain_raw as varchar))), '^(www|mail)\.', '') as company_domain_clean,
    job_title_text as job_title_clean,
    company_name_messy is null as company_name_missing,
    company_domain_raw is null as company_domain_missing,
    job_title_raw is null as job_title_missing,
    regexp_replace(marketing_channel_text, '[\s-]+', '_', 'g') as marketing_channel_clean,
    campaign_tier_text as campaign_tier_clean,
    case when region_text = 'UNKNOWN' then null else region_text end as region_clean,
    not (regexp_replace(marketing_channel_text, '[\s-]+', '_', 'g') in (
        'paid_search', 'organic', 'linkedin_ads', 'webinar', 'partner', 'event', 'email', 'direct', 'other'
    )) as marketing_channel_unmapped,
    not (campaign_tier_text in (
        'tier1_enterprise', 'tier2_growth', 'tier3_smb', 'pilot', 'legacy_2019', 'sunset'
    )) as campaign_tier_unmapped,
    not coalesce((region_text in ('NA', 'EMEA', 'APAC', 'LATAM')), false) as region_unmapped,
    case
        when reply_text in ('yes', 'y') then true
        when reply_text in ('no', 'n') then false
        else null
    end as replied_within_7d_bool,
    reply_text is null as replied_within_7d_missing,
    employee_band_text as employee_band_normalized,
    coalesce(employee_band_text in ('unknown', 'n/a', 'seelinkedin'), true) as employee_band_missing_or_unknown,
    coalesce(
        try_cast(regexp_extract(employee_band_text, '^(\d+)-\d+$', 1) as bigint),
        try_cast(regexp_extract(employee_band_text, '^(\d+)\+$', 1) as bigint),
        try_cast(regexp_extract(employee_band_text, '^~(\d+)$', 1) as bigint),
        try_cast(regexp_extract(employee_band_text, '^(\d+)$', 1) as bigint)
    ) as employee_count_lower,
    coalesce(
        try_cast(regexp_extract(employee_band_text, '^\d+-(\d+)$', 1) as bigint),
        try_cast(regexp_extract(employee_band_text, '^~(\d+)$', 1) as bigint),
        try_cast(regexp_extract(employee_band_text, '^(\d+)$', 1) as bigint)
    ) as employee_count_upper,
    case
        when coalesce(
            try_cast(regexp_extract(employee_band_text, '^(\d+)-\d+$', 1) as bigint),
            try_cast(regexp_extract(employee_band_text, '^(\d+)\+$', 1) as bigint),
            try_cast(regexp_extract(employee_band_text, '^~(\d+)$', 1) as bigint),
            try_cast(regexp_extract(employee_band_text, '^(\d+)$', 1) as bigint)
        ) is null then null
        when coalesce(
            try_cast(regexp_extract(employee_band_text, '^(\d+)-\d+$', 1) as bigint),
            try_cast(regexp_extract(employee_band_text, '^(\d+)\+$', 1) as bigint),
            try_cast(regexp_extract(employee_band_text, '^~(\d+)$', 1) as bigint),
            try_cast(regexp_extract(employee_band_text, '^(\d+)$', 1) as bigint)
        ) >= 5000 then 'enterprise'
        when coalesce(
            try_cast(regexp_extract(employee_band_text, '^\d+-(\d+)$', 1) as bigint),
            try_cast(regexp_extract(employee_band_text, '^~(\d+)$', 1) as bigint),
            try_cast(regexp_extract(employee_band_text, '^(\d+)$', 1) as bigint)
        ) <= 50 then 'small'
        when coalesce(
            try_cast(regexp_extract(employee_band_text, '^\d+-(\d+)$', 1) as bigint),
            try_cast(regexp_extract(employee_band_text, '^~(\d+)$', 1) as bigint),
            try_cast(regexp_extract(employee_band_text, '^(\d+)$', 1) as bigint)
        ) <= 500 then 'mid_market'
        else 'commercial'
    end as employee_size_bucket,
    web_session_seconds is null as web_session_seconds_missing,
    coalesce(web_session_seconds > 86400, false) as web_session_seconds_outlier,
    case
        when web_session_seconds is null then null
        when web_session_seconds < 0 then 0.0
        when web_session_seconds > 86400 then 86400.0
        else web_session_seconds
    end as web_session_seconds_clean,
    lead_score is null as lead_score_missing,
    case
        when lead_score is null then null
        when lead_score < 0 then 0.0
        when lead_score > 100 then 100.0
        else lead_score
    end as lead_score_clean,
    cast(demo_requested as boolean) as demo_requested_bool
from normalized
