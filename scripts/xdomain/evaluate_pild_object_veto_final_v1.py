#!/usr/bin/env python3
"""Object-level physical review used in the paper (final configuration).

Decision unit: connected components of the visual mask.
Descriptor: terrain geometry, confidence, spectral change, and hydro-topology features.
Source one-hot is included because the source identity is known at deployment time;
events remain fully isolated by GroupKFold.
Learner: HistGradientBoostingRegressor purity model with a five-seed ensemble.
Rule: veto when predicted purity < IoU_base / (1 + IoU_base).

This configuration targets transfer to new events from known sources. It does not claim
zero-shot transfer to an entirely unseen data source.
"""

from __future__ import annotations 

import argparse 
import json 
import time 
from pathlib import Path 

import numpy as np 
import pandas as pd 
from sklearn .ensemble import HistGradientBoostingRegressor 
from sklearn .model_selection import GroupKFold 

SCRIPT_DIR =Path (__file__ ).resolve ().parent 
PROJECT_ROOT =SCRIPT_DIR .parents [1 ]
DEFAULT_UNITS =PROJECT_ROOT /"experiments/revision2026/pild_subobject_units_v1"
DEFAULT_HYDRO =(
PROJECT_ROOT 
/"experiments/revision2026/pild_object_hydrology_features_v1/object_hydrology_features.parquet"
)

CONFIDENCE =["mean_probability","max_probability","p90_probability"]
TERRAIN =[
"area_px","log_area","mean_slope","p10_slope","p90_slope","flat_fraction",
"steep_fraction","elev_range","relative_relief","aspect_coherence","elongation",
"downslope_alignment","descent_consistency","slope_decline","divide_straddle",
"tpi900_range","mean_tpi_90m","mean_tpi_300m","mean_tpi_900m",
"valley_bottom_fraction","mean_valley_depth","mean_ridge_height","mean_ruggedness",
"mean_local_relief_300m","mean_plan_curvature","mean_profile_curvature","compactness",
]
REFERENCE_IOU =0.21819164482792633 


def ensemble_oof (x ,y ,groups ,seeds ,n_splits ,max_iter ,max_leaf_nodes ):
    total =np .zeros (len (y ),dtype =float )
    for seed in seeds :
        out =np .zeros (len (y ),dtype =float )
        for train ,test in GroupKFold (n_splits ).split (x ,y ,groups =groups ):
            model =HistGradientBoostingRegressor (
            max_iter =max_iter ,max_leaf_nodes =max_leaf_nodes ,learning_rate =0.06 ,
            l2_regularization =1.0 ,random_state =seed ,
            )
            model .fit (x [train ],y [train ])
            out [test ]=model .predict (x [test ])
        total +=np .clip (out ,0.0 ,1.0 )
    return total /len (seeds )


def pooled_outcome (remove ,frame ,tp ,fp ,fn ,base_iou ):
    i_px =frame .intersection_px .to_numpy (dtype =float )
    f_px =frame .false_px .to_numpy (dtype =float )
    lost =float (i_px [remove ].sum ())
    cleared =float (f_px [remove ].sum ())
    purity =frame .purity .to_numpy ()
    base_err =fp +fn 
    new_err =(fp -cleared )+(fn +lost )
    return {
    "n_units":int (len (frame )),
    "n_removed":int (remove .sum ()),
    "delta_iou":float ((tp -lost )/(tp +fp +fn -cleared )-base_iou ),
    "iou_after":float ((tp -lost )/(tp +fp +fn -cleared )),
    "rer":float ((base_err -new_err )/base_err ),
    "lost_tp":lost ,
    "cleared_fp":cleared ,
    "fp_mass_captured":float (cleared /fp ),
    "tp_mass_lost":float (lost /tp ),
    "corrected_to_harmed":float (cleared /max (lost ,1.0 )),
    "removal_precision":float (
    1.0 -(remove &(purity >=base_iou )).sum ()/max (remove .sum (),1 )
    ),
    }


def per_group_delta (remove ,frame ,base_iou ,key ):
    work =frame .assign (_rm =remove )
    rows =[]
    for name ,block in work .groupby (key ):
        e_tp =float (block .intersection_px .sum ())
        e_fp =float (block .false_px .sum ())
        if e_tp <=0 :
            continue 
        e_fn =max (e_tp /base_iou -e_tp -e_fp ,0.0 )
        lost =float (block .intersection_px [block ._rm ].sum ())
        cleared =float (block .false_px [block ._rm ].sum ())
        denom =e_tp +e_fp +e_fn 
        base =e_tp /denom 
        after =(e_tp -lost )/max (denom -cleared ,1.0 )
        base_err =e_fp +e_fn 
        new_err =(e_fp -cleared )+(e_fn +lost )
        rows .append (
        {
        key :name ,
        "n_units":int (len (block )),
        "delta_iou":after -base ,
        "rer":(base_err -new_err )/base_err if base_err >0 else 0.0 ,
        }
        )
    return pd .DataFrame (rows )


def bootstrap_ci (values ,n_boot =5000 ,seed =0 ):
    rng =np .random .default_rng (seed )
    draws =rng .integers (0 ,len (values ),size =(n_boot ,len (values )))
    means =np .asarray (values )[draws ].mean (axis =1 )
    return float (np .percentile (means ,2.5 )),float (np .percentile (means ,97.5 ))


def main ()->None :
    parser =argparse .ArgumentParser ()
    parser .add_argument ("--units",type =Path ,default =DEFAULT_UNITS )
    parser .add_argument ("--hydrology",type =Path ,default =DEFAULT_HYDRO )
    parser .add_argument ("--seeds",type =int ,nargs ="+",default =[0 ,7 ,101 ,2029 ,55555 ])
    parser .add_argument ("--n-splits",type =int ,default =5 )
    parser .add_argument ("--max-iter",type =int ,default =800 )
    parser .add_argument ("--max-leaf-nodes",type =int ,default =63 )
    parser .add_argument (
    "--baseline-iou",type =float ,default =REFERENCE_IOU ,
    help ="Pooled baseline IoU of this visual anchor; change when the anchor changes",
    )
    parser .add_argument (
    "--anchor",default ="prithvi_eo2_300m_tl",
    help ="Visual-anchor tag used only for artifact bookkeeping",
    )
    parser .add_argument (
    "--outdir",type =Path ,
    default =PROJECT_ROOT /"experiments/revision2026/pild_object_veto_final_v1",
    )
    args =parser .parse_args ()
    started =time .time ()

    whole =pd .read_parquet (args .units /"units_whole.parquet")
    hydro =pd .read_parquet (args .hydrology )
    frame =whole .merge (hydro ,on =["sample_id","component_id"],how ="inner").reset_index (drop =True )
    if len (frame )!=len (whole ):
        raise RuntimeError (f"join failed: {len (whole )} -> {len (frame )}")

    spec_cols =[c for c in frame .columns if c .startswith ("spec_")]
    hyd_cols =[c for c in frame .columns if c .startswith ("hyd_")]
    feature_cols =TERRAIN +CONFIDENCE +spec_cols +hyd_cols 

    tp =float (frame .intersection_px .sum ())
    fp =float (frame .false_px .sum ())
    base_iou =float (args .baseline_iou )
    fn =tp /base_iou -tp -fp 
    cut =base_iou /(1.0 +base_iou )

    y =frame .purity .to_numpy (dtype =float )
    groups =frame .canonical_event_id .to_numpy ()
    source_onehot =pd .get_dummies (frame .dataset_id ).to_numpy (dtype =float )
    x_plain =frame [feature_cols ].to_numpy (dtype =float )
    x_source =np .hstack ([x_plain ,source_onehot ])

    print (
    f"units={len (frame ):,} events={frame .canonical_event_id .nunique ()}  "
    f" {len (feature_cols )} (terrain={len (TERRAIN )} / confidence={len (CONFIDENCE )} / "
    f" {len (spec_cols )} / hydro={len (hyd_cols )})"
    )
    print (f"pooled TP={tp :,.0f} FP={fp :,.0f} FN={fn :,.0f}  baseline IoU={base_iou :.5f}  cut={cut :.5f}\n")

    results ={}
    for name ,x in (("source_conditioned",x_source ),("source_blind",x_plain )):
        score =ensemble_oof (
        x ,y ,groups ,args .seeds ,args .n_splits ,args .max_iter ,args .max_leaf_nodes 
        )
        remove =score <cut 
        outcome =pooled_outcome (remove ,frame ,tp ,fp ,fn ,base_iou )
        events =per_group_delta (remove ,frame ,base_iou ,"canonical_event_id")
        sources =per_group_delta (remove ,frame ,base_iou ,"dataset_id")
        d_lo ,d_hi =bootstrap_ci (events .delta_iou .to_numpy ())
        r_lo ,r_hi =bootstrap_ci (events .rer .to_numpy ())
        outcome .update (
        {
        "spearman":float (pd .Series (score ).corr (pd .Series (y ),method ="spearman")),
        "event_macro_delta_iou":float (events .delta_iou .mean ()),
        "event_macro_delta_ci":[d_lo ,d_hi ],
        "event_macro_rer":float (events .rer .mean ()),
        "event_macro_rer_ci":[r_lo ,r_hi ],
        "events_positive":int ((events .delta_iou >0 ).sum ()),
        "n_events":int (len (events )),
        "source_macro_delta_iou":float (sources .delta_iou .mean ()),
        }
        )
        results [name ]={"outcome":outcome ,"events":events ,"sources":sources ,"score":score }
        print (
        f"[{name }]\n"
        f"  ΔIoU={outcome ['delta_iou']:+.5f}  IoU {base_iou :.5f} -> {outcome ['iou_after']:.5f}\n"
        f"  RER={outcome ['rer']:+.4f}  corrected/harmed={outcome ['corrected_to_harmed']:.2f}  "
        f"={outcome ['removal_precision']:.3f}\n"
        f"  FP  {outcome ['fp_mass_captured']:.1%}，TP  {outcome ['tp_mass_lost']:.1%}\n"
        f"   ΔIoU={outcome ['event_macro_delta_iou']:+.5f} [{d_lo :+.5f}, {d_hi :+.5f}]  "
        f"RER={outcome ['event_macro_rer']:+.4f} [{r_lo :+.4f}, {r_hi :+.4f}]  "
        f"{outcome ['events_positive']}/{outcome ['n_events']} \n"
        f"   ΔIoU={outcome ['source_macro_delta_iou']:+.5f}"
        )
        print ("text")
        for _ ,row in sources .iterrows ():
            print (
            f"    {row .dataset_id :26s} ΔIoU={row .delta_iou :+.5f}  RER={row .rer :+.4f}  "
            f"n={int (row .n_units ):,}"
            )
        print ()

    args .outdir .mkdir (parents =True ,exist_ok =True )
    for name ,payload in results .items ():
        payload ["events"].to_csv (args .outdir /f"{name }_by_event.csv",index =False )
        payload ["sources"].to_csv (args .outdir /f"{name }_by_source.csv",index =False )
    frame_out =frame [["sample_id","dataset_id","canonical_event_id","component_id",
    "purity","intersection_px","false_px","area_px"]].copy ()
    frame_out ["score_source_conditioned"]=results ["source_conditioned"]["score"]
    frame_out ["score_source_blind"]=results ["source_blind"]["score"]
    frame_out ["removed"]=frame_out .score_source_conditioned <cut 
    frame_out .to_parquet (args .outdir /"component_decisions.parquet",index =False )

    verdict ={
    "delta_iou":results ["source_conditioned"]["outcome"]["delta_iou"],
    "rer":results ["source_conditioned"]["outcome"]["rer"],
    "source_conditioning_gain":(
    results ["source_conditioned"]["outcome"]["delta_iou"]
    -results ["source_blind"]["outcome"]["delta_iou"]
    ),
    "reaches_target_delta_iou":bool (
    results ["source_conditioned"]["outcome"]["delta_iou"]>=0.03 
    ),
    "reaches_target_rer":bool (results ["source_conditioned"]["outcome"]["rer"]>=0.10 ),
    }
    print ("text"+json .dumps (verdict ,ensure_ascii =False ,indent =2 ))
    (args .outdir /"summary.json").write_text (
    json .dumps (
    {
    "schema_version":"pild_object_veto_final.v1",
    "evidence_status":"event-grouped OOF on the paper folds",
    "visual_anchor":args .anchor ,
    "disclosure":(
    "Source-conditioned features target new events from known sources; "
    "transfer to a fully unseen source is weaker "
    "(Spearman 0.242 / ΔIoU +0.0006 in the held-out source protocol)."
    ),
    "baseline_iou":base_iou ,
    "analytic_cut":cut ,
    "seeds":args .seeds ,
    "feature_groups":{
    "terrain":len (TERRAIN ),
    "confidence":len (CONFIDENCE ),
    "spectral":len (spec_cols ),
    "hydrology":len (hyd_cols ),
    },
    "results":{k :v ["outcome"]for k ,v in results .items ()},
    "verdict":verdict ,
    "elapsed_seconds":round (time .time ()-started ,2 ),
    },
    ensure_ascii =False ,
    indent =2 ,
    ),
    encoding ="utf-8",
    )
    print (f"\n {args .outdir }")


if __name__ =="__main__":
    main ()
