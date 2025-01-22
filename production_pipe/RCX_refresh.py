from delta import *
from pyspark.sql import functions as F
import os
from utils import *
from datetime import datetime

def load_src():
    la = spark.read.format("delta").load("dbfs:/mnt/edhprodenrich/Enterprise Data Store/Secured/data/Cleanse/RCX/RCX_loyalty_activities/CurrentState")
    lai = spark.read.format("delta").load('dbfs:/mnt/edhprodenrich/Enterprise Data Store/Secured/data/Cleanse/RCX/RCX_loyalty_accrual_items/CurrentState')
    MLI = spark.read.format("delta").load("dbfs:/mnt/edhprodenrich/Enterprise Data Store/Secured/data/Cleanse/RCX/RCX_member_loyalty_ids/CurrentState")
    PP = spark.read.format("delta").load('dbfs:/mnt/edhprodenrich/Enterprise Data Store/Secured/data/Cleanse/RCX/RCX_purse_policies/CurrentState')
    l = spark.read.format("delta").load('dbfs:/mnt/edhprodenrich/Enterprise Data Store/Secured/data/Cleanse/RCX/RCX_locations/CurrentState')

    la.createOrReplaceTempView("la")
    lai.createOrReplaceTempView("lai")
    MLI.createOrReplaceTempView('MLI')
    PP.createOrReplaceTempView('PP')
    l.createOrReplaceTempView('l')

    m = spark.read.format("delta").load("dbfs:/mnt/edhprodenrich/Enterprise Data Store/Secured/data/Cleanse/RCX/RCX_members/CurrentState")
    mt = spark.read.format("delta").load('dbfs:/mnt/edhprodenrich/Enterprise Data Store/Secured/data/Cleanse/RCX/RCX_member_tiers/CurrentState')
    mth = spark.read.format("delta").load('dbfs:/mnt/edhprodenrich/Enterprise Data Store/Secured/data/Cleanse/RCX/RCX_member_tier_histories/CurrentState')
    merged = spark.read.format("delta").load('dbfs:/mnt/edhprodenrich/Enterprise Data Store/Secured/data/Cleanse/RCX/RCX_merged_members/CurrentState')

    m.createOrReplaceTempView("m")
    mt.createOrReplaceTempView("mt")
    mth.createOrReplaceTempView("mth")
    merged.createOrReplaceTempView("merged")

    return

def load_tc(year):
    tc_hist = spark.sql(f'''
    SELECT  
        mli.loyalty_id AS playerid,
        CASE WHEN la.activity_date BETWEEN '2016-03-13 01:59:59.999999' AND '2016-11-06 01:59:59.999999' THEN la.activity_date  - INTERVAL '7'  HOUR
        WHEN la.activity_date BETWEEN '2017-03-12 01:59:59.999999' AND '2017-11-05 01:59:59.999999' THEN la.activity_date  - INTERVAL '7'  HOUR
        WHEN la.activity_date BETWEEN '2018-03-11 01:59:59.999999' AND '2018-11-04 01:59:59.999999' THEN la.activity_date  - INTERVAL '7'  HOUR
        WHEN la.activity_date BETWEEN '2019-03-10 01:59:59.999999' AND '2019-11-03 01:59:59.999999' THEN la.activity_date  - INTERVAL '7'  HOUR
        WHEN la.activity_date BETWEEN '2020-03-10 01:59:59.999999' AND '2020-11-01 01:59:59.999999' THEN la.activity_date  - INTERVAL '7'  HOUR
        WHEN la.activity_date BETWEEN '2021-03-14 01:59:59.999999' AND '2021-11-07 01:59:59.999999' THEN la.activity_date  - INTERVAL '7'  HOUR
        WHEN la.activity_date BETWEEN '2022-03-13 01:59:59.999999' AND '2022-11-06 01:59:59.999999' THEN la.activity_date  - INTERVAL '7'  HOUR
        WHEN la.activity_date BETWEEN '2023-03-12 01:59:59.999999' AND '2023-11-05 01:59:59.999999' THEN la.activity_date  - INTERVAL '7'  HOUR
        WHEN la.activity_date BETWEEN '2024-03-10 01:59:59.999999' AND '2024-11-03 01:59:59.999999' THEN la.activity_date  - INTERVAL '7'  HOUR
        WHEN la.activity_date BETWEEN '2025-03-09 01:59:59.999999' AND '2025-11-02 01:59:59.999999' THEN la.activity_date  - INTERVAL '7'  HOUR
        ELSE la.activity_date  - INTERVAL '8' HOUR end AS transactiondate, --conversion from UTC to Pacific timezone
        CAST (lai.accrued_pts/10000.0000 AS DECIMAL(18,4)) AS tiercredit, l.site_name,
        case when la.lob LIKE 'Gaming%' then 'Gaming'
        when la.user_id like 'RCX%' then 'Non-Gaming'
        when la.lob in ('BetMGM','M life Credit Card','Other') then 'Other'
        when la.lob in ('Coffee','Entertainment','FB','Hotel','Spa') then 'Non-Gaming'
        else 'Other' end as lob_rollup
    FROM la
    LEFT JOIN lai
    ON      la.id = lai.loyalty_activity_id
    left join l on la.location_id = l.id and l.delete_flag = 'N'
    LEFT JOIN mli
    ON      la.member_id = mli.member_id
    LEFT JOIN pp
    ON      lai.purse_policy_id = pp.id
    WHERE   policy_name like 'Tier Credits%'
    AND     activity_status NOT IN ('Error')
    AND     la.delete_flag = 'N'
    AND     lai.delete_flag = 'N'
    AND     mli.delete_flag = 'N'
    AND     pp.delete_flag = 'N'
    --AND     la.eff_end_dttm = DATE '9999-12-31'
    --AND     lai.eff_end_dttm = DATE '9999-12-31'
    --AND     mli.eff_end_dttm = DATE '9999-12-31'
    --AND     pp.eff_end_dttm = DATE '9999-12-31'
    AND     mli.loyalty_id_name = 'PlayerId'
    AND CASE WHEN la.activity_date BETWEEN '2016-03-13 01:59:59.999999' AND '2016-11-06 01:59:59.999999' THEN la.activity_date  - INTERVAL '7'  HOUR
        WHEN la.activity_date BETWEEN '2017-03-12 01:59:59.999999' AND '2017-11-05 01:59:59.999999' THEN la.activity_date  - INTERVAL '7'  HOUR
        WHEN la.activity_date BETWEEN '2018-03-11 01:59:59.999999' AND '2018-11-04 01:59:59.999999' THEN la.activity_date  - INTERVAL '7'  HOUR
        WHEN la.activity_date BETWEEN '2019-03-10 01:59:59.999999' AND '2019-11-03 01:59:59.999999' THEN la.activity_date  - INTERVAL '7'  HOUR
        WHEN la.activity_date BETWEEN '2020-03-10 01:59:59.999999' AND '2020-11-01 01:59:59.999999' THEN la.activity_date  - INTERVAL '7'  HOUR
        WHEN la.activity_date BETWEEN '2021-03-14 01:59:59.999999' AND '2021-11-07 01:59:59.999999' THEN la.activity_date  - INTERVAL '7'  HOUR
        WHEN la.activity_date BETWEEN '2022-03-13 01:59:59.999999' AND '2022-11-06 01:59:59.999999' THEN la.activity_date  - INTERVAL '7'  HOUR
        WHEN la.activity_date BETWEEN '2023-03-12 01:59:59.999999' AND '2023-11-05 01:59:59.999999' THEN la.activity_date  - INTERVAL '7'  HOUR
        WHEN la.activity_date BETWEEN '2024-03-10 01:59:59.999999' AND '2024-11-03 01:59:59.999999' THEN la.activity_date  - INTERVAL '7'  HOUR
        WHEN la.activity_date BETWEEN '2025-03-09 01:59:59.999999' AND '2025-11-02 01:59:59.999999' THEN la.activity_date  - INTERVAL '7'  HOUR
        ELSE la.activity_date  - INTERVAL '8' HOUR end between date'{year}-01-01' and date'{year}-12-31'
        ''')
    
    return tc_hist


def load_tier_promo():
    # RW/KM 5/17/24   for ticket LC-1641
    # This pre-view was a cte named cte on original code
    # this considers players who DO NOT HAVE any tier history to them, hence tier before and after will be 'NA' and sync issues will be 'N'
    cte1 = spark.sql(
        '''
        select  m.id as rcx_id, 
                mli.loyalty_id as PlayerID,
                enrollment_date, 
                coalesce(prev_tier_level,level_name) as tier_before_change, 
                coalesce(current_tier_level,level_name) as tier_after_change,  
                coalesce(assign_date,achieved_on_utc_ts) as change_assigned, 
                coalesce(level_name, 'NA') as current_tier,
                coalesce(achieved_on_utc_ts, to_timestamp('9999-12-31 00:00:00.000000')) as current_tier_achieved,
                mt.reason as mt_reason, mt.sub_reason as mt_subreason, mth.reason as mth_reason, mth.sub_reason as mth_subreason
        from    m 
        join    mt
        on      m.id = mt.member_id
        left join mth
        on      mth.member_id = m.id and mth.delete_flag = 'N' --and	mth.eff_end_dttm = DATE '9999-12-31'
        left join mli
        on      mli.member_id = m.id 
        where   m.delete_flag = 'N'
        and     mt.delete_flag = 'N' 
        and     mli.loyalty_id_name = 'PlayerId'
        and 	mli.delete_flag = 'N'
        --and		m.eff_end_dttm = DATE '9999-12-31'
        --and		mt.eff_end_dttm = DATE '9999-12-31'
        --and		mli.eff_end_dttm = DATE '9999-12-31'
        and     m.id  not in --37,853,142 || 33,326,418
        (
        select  distinct member_id
        from    mth
        where	delete_flag = 'N'
        --and		eff_end_dttm = DATE '9999-12-31'
        )

        UNION

        select  m.id as rcx_id, --this considers players who DO HAVE tier history to them, hence tier before and after will be '(blank)' for null and sync issues will be 'Y'
                mli.loyalty_id as PlayerID,
                enrollment_date, 
                coalesce(prev_tier_level,'Sapphire') as tier_before_change, 
                coalesce(current_tier_level,prev_tier_level, 'Sapphire') as tier_after_change, 
                coalesce(assign_date,achieved_on_utc_ts) as change_assigned, 
                coalesce(level_name, 'NA') as current_tier,
                coalesce(achieved_on_utc_ts, to_timestamp('9999-12-31 00:00:00.000000')) as current_tier_achieved,
                mt.reason as mt_reason, mt.sub_reason as mt_subreason, mth.reason as mth_reason, mth.sub_reason as mth_subreason
        from    m 
        join    mt
        on      m.id = mt.member_id
        left join mth
        on      mth.member_id = m.id and mth.delete_flag = 'N'
        left join mli
        on      mli.member_id = m.id 
        where   m.delete_flag = 'N'
        and     mt.delete_flag = 'N' -- 71166190
        and     mli.loyalty_id_name = 'PlayerId'
        and 	mli.delete_flag = 'N'
        --and		mth.eff_end_dttm = DATE '9999-12-31'
        --and		m.eff_end_dttm = DATE '9999-12-31'
        --and		mt.eff_end_dttm = DATE '9999-12-31'
        --and		mli.eff_end_dttm = DATE '9999-12-31'
        and     m.id in 
        (
        select  distinct member_id 
        from    mth
        where	mth.delete_flag = 'N'
        --and		eff_end_dttm = DATE '9999-12-31'
    )
    '''
    )

    cte1.createOrReplaceTempView("cte1")

    cte2 = spark.sql(
        '''
        select  rcx_id,
            PlayerID,
            enrollment_date, 
            tier_before_change, 
            tier_after_change, 
            change_assigned, 
            current_tier,
            current_tier_achieved,
            case  
                when coalesce(survivor_id, 'NA') = 'NA' then rcx_id
                else  survivor_id
            end as  survivor_id,
            mt_reason, mt_subreason, mth_reason, mth_subreason
    from    cte1 c
    left join merged
    on      c.rcx_id = merged.victim_id and merged.delete_flag = 'N' --and m.eff_end_dttm = DATE '9999-12-31'
    '''
    )

    cte2.createOrReplaceTempView('cte2')

    Tier_History = spark.sql(
        '''
        select  cast(PlayerID as varchar(20)) as playerid,
                --enrollment_date, 
                CASE WHEN enrollment_date BETWEEN '2016-03-13 01:59:59.999999' AND '2016-11-06 01:59:59.999999' THEN enrollment_date  - INTERVAL '7'  HOUR
            WHEN enrollment_date BETWEEN '2017-03-12 01:59:59.999999' AND '2017-11-05 01:59:59.999999' THEN enrollment_date  - INTERVAL '7'  HOUR
            WHEN enrollment_date BETWEEN '2018-03-11 01:59:59.999999' AND '2018-11-04 01:59:59.999999' THEN enrollment_date  - INTERVAL '7'  HOUR
            WHEN enrollment_date BETWEEN '2019-03-10 01:59:59.999999' AND '2019-11-03 01:59:59.999999' THEN enrollment_date  - INTERVAL '7'  HOUR
            WHEN enrollment_date BETWEEN '2020-03-10 01:59:59.999999' AND '2020-11-01 01:59:59.999999' THEN enrollment_date  - INTERVAL '7'  HOUR
            WHEN enrollment_date BETWEEN '2021-03-14 01:59:59.999999' AND '2021-11-07 01:59:59.999999' THEN enrollment_date  - INTERVAL '7'  HOUR
            WHEN enrollment_date BETWEEN '2022-03-13 01:59:59.999999' AND '2022-11-06 01:59:59.999999' THEN enrollment_date  - INTERVAL '7'  HOUR
            WHEN enrollment_date BETWEEN '2023-03-12 01:59:59.999999' AND '2023-11-05 01:59:59.999999' THEN enrollment_date  - INTERVAL '7'  HOUR
            WHEN enrollment_date BETWEEN '2024-03-10 01:59:59.999999' AND '2024-11-03 01:59:59.999999' THEN enrollment_date  - INTERVAL '7'  HOUR
            WHEN enrollment_date BETWEEN '2025-03-09 01:59:59.999999' AND '2025-11-02 01:59:59.999999' THEN enrollment_date  - INTERVAL '7'  HOUR
        ELSE enrollment_date  - INTERVAL '8' HOUR end AS enrollment_date,
                tier_before_change, 
                tier_after_change,
                CASE WHEN change_assigned BETWEEN '2016-03-13 01:59:59.999999' AND '2016-11-06 01:59:59.999999' THEN change_assigned  - INTERVAL '7'  HOUR
            WHEN change_assigned BETWEEN '2017-03-12 01:59:59.999999' AND '2017-11-05 01:59:59.999999' THEN change_assigned  - INTERVAL '7'  HOUR
            WHEN change_assigned BETWEEN '2018-03-11 01:59:59.999999' AND '2018-11-04 01:59:59.999999' THEN change_assigned  - INTERVAL '7'  HOUR
            WHEN change_assigned BETWEEN '2019-03-10 01:59:59.999999' AND '2019-11-03 01:59:59.999999' THEN change_assigned  - INTERVAL '7'  HOUR
            WHEN change_assigned BETWEEN '2020-03-10 01:59:59.999999' AND '2020-11-01 01:59:59.999999' THEN change_assigned  - INTERVAL '7'  HOUR
            WHEN change_assigned BETWEEN '2021-03-14 01:59:59.999999' AND '2021-11-07 01:59:59.999999' THEN change_assigned  - INTERVAL '7'  HOUR
            WHEN change_assigned BETWEEN '2022-03-13 01:59:59.999999' AND '2022-11-06 01:59:59.999999' THEN change_assigned  - INTERVAL '7'  HOUR
            WHEN change_assigned BETWEEN '2023-03-12 01:59:59.999999' AND '2023-11-05 01:59:59.999999' THEN change_assigned  - INTERVAL '7'  HOUR
            WHEN change_assigned BETWEEN '2024-03-10 01:59:59.999999' AND '2024-11-03 01:59:59.999999' THEN change_assigned  - INTERVAL '7'  HOUR
            WHEN change_assigned BETWEEN '2025-03-09 01:59:59.999999' AND '2025-11-02 01:59:59.999999' THEN change_assigned  - INTERVAL '7'  HOUR
        ELSE change_assigned  - INTERVAL '8' HOUR end AS change_assigned,
                current_tier,
            --    current_tier_achieved,
            CASE WHEN current_tier_achieved BETWEEN '2016-03-13 01:59:59.999999' AND '2016-11-06 01:59:59.999999' THEN current_tier_achieved  - INTERVAL '7'  HOUR
            WHEN current_tier_achieved BETWEEN '2017-03-12 01:59:59.999999' AND '2017-11-05 01:59:59.999999' THEN current_tier_achieved  - INTERVAL '7'  HOUR
            WHEN current_tier_achieved BETWEEN '2018-03-11 01:59:59.999999' AND '2018-11-04 01:59:59.999999' THEN current_tier_achieved  - INTERVAL '7'  HOUR
            WHEN current_tier_achieved BETWEEN '2019-03-10 01:59:59.999999' AND '2019-11-03 01:59:59.999999' THEN current_tier_achieved  - INTERVAL '7'  HOUR
            WHEN current_tier_achieved BETWEEN '2020-03-10 01:59:59.999999' AND '2020-11-01 01:59:59.999999' THEN current_tier_achieved  - INTERVAL '7'  HOUR
            WHEN current_tier_achieved BETWEEN '2021-03-14 01:59:59.999999' AND '2021-11-07 01:59:59.999999' THEN current_tier_achieved  - INTERVAL '7'  HOUR
            WHEN current_tier_achieved BETWEEN '2022-03-13 01:59:59.999999' AND '2022-11-06 01:59:59.999999' THEN current_tier_achieved  - INTERVAL '7'  HOUR
            WHEN current_tier_achieved BETWEEN '2023-03-12 01:59:59.999999' AND '2023-11-05 01:59:59.999999' THEN current_tier_achieved  - INTERVAL '7'  HOUR
            WHEN current_tier_achieved BETWEEN '2024-03-10 01:59:59.999999' AND '2024-11-03 01:59:59.999999' THEN current_tier_achieved  - INTERVAL '7'  HOUR
            WHEN current_tier_achieved BETWEEN '2025-03-09 01:59:59.999999' AND '2025-11-02 01:59:59.999999' THEN current_tier_achieved  - INTERVAL '7'  HOUR
        ELSE current_tier_achieved  - INTERVAL '8' HOUR end AS current_tier_achieved,
            cast(mli.loyalty_id as bigint) as survivorId, mt_reason, mt_subreason, mth_reason, mth_subreason
        from     cte2 c2
        left join mli
        -- on      mli.member_id = cast(cte2.survivor_id as varchar(15))
        on      mli.member_id = c2.survivor_id 
        where   mli.loyalty_id_name = 'PlayerId'
        and 	mli.delete_flag = 'N'
        --and		mli.eff_end_dttm = DATE '9999-12-31'
        '''
    )

    return Tier_History



if __name__ == "__main__":
    load_src()
    output_path = '/mnt/proddatalake/dev/RCX/'

    TC = load_tc(2025)
    TC_path = output_path+"TC_2025.parquet"
    TC.write.mode("overwrite").parquet(TC_path)

    write_log(f'[{datetime.utcnow()}] STEP [1] - Tier Credits Earnings has been refreshed, latest:{TC.select(F.max("transactiondate")).collect()[0][0]}')

    Tier_pro_path = output_path+"Tier_History.parquet"
    Tier_pro = load_tier_promo()
    Tier_pro.write.mode("overwrite").parquet(Tier_pro_path)

    write_log(f'[{datetime.utcnow()}] STEP [2] - Tier Promotions Histories has been refreshed, latest:{Tier_pro.select(F.max("change_assigned")).collect()[0][0]}')



    