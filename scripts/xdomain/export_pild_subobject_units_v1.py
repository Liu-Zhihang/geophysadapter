#!/usr/bin/env python3
"""Split connected visual components into geomorphically consistent sub-objects.

Boundary cues after robust within-sample normalization:
divide (plan curvature), aspect_break, and slope_break around a critical angle.
With material support the critical angle is:
theta_c = 22 + 16 * rank01(z(sand) + 0.5 z(cfvo) - z(clay) - 0.5 z(awc));
otherwise it falls back to 25 degrees.
Export modes: whole, geomorphic, material, material_shuffled.
"""

from __future__ import annotations 

import argparse 
import json 
import sys 
import time 
from pathlib import Path 

import numpy as np 
import pandas as pd 
from scipy import ndimage 
from skimage .segmentation import watershed 

SCRIPT_DIR =Path (__file__ ).resolve ().parent 
PROJECT_ROOT =SCRIPT_DIR .parents [1 ]
sys .path .insert (0 ,str (SCRIPT_DIR ))

from analyze_pild_object_physical_separability_v1 import (# noqa: E402
TERRAIN_INDEX ,
component_features ,
)
from export_pild_object_spectral_features_v1 import (# noqa: E402
component_spectral_features ,
spectral_index_stack ,
)

DEFAULT_CACHE =(
PROJECT_ROOT /"experiments/revision2026/pild_object_physical_diagnostic_v1/oof_cache"
)
FOLD_IDS =[f"source_stratified_{i }"for i in range (4 )]

# ROLE_MATERIAL_FEATURE_NAMES column order: five AWC depths, then (clay,sand,silt,cec,soc,bdod,cfvo,phh2o) x (0_5cm,5_15cm)
AWC_SLICE =slice (0 ,5 )
CLAY_IDX =(5 ,6 )
SAND_IDX =(7 ,8 )
CFVO_IDX =(17 ,18 )

THETA_MIN_DEG =22.0 
THETA_SPAN_DEG =16.0 
THETA_FALLBACK_DEG =25.0 
SLOPE_BREAK_WIDTH_DEG =3.0 
MARKER_QUANTILE =0.35 
MIN_AREA =4 


def robust_unit (values :np .ndarray ,mask :np .ndarray )->np .ndarray :
    """Robust normalize by the in-mask 95th percentile; clip to [0, 1]."""
    if not mask .any ():
        return np .zeros_like (values ,dtype =np .float32 )
    scale =float (np .percentile (np .abs (values [mask ]),95 ))
    if scale <1e-6 :
        return np .zeros_like (values ,dtype =np .float32 )
    return np .clip (values /scale ,0.0 ,1.0 ).astype (np .float32 )


def rank01 (values :np .ndarray )->np .ndarray :
    """Rank-normalize to [0, 1] using features only; no labels."""
    order =np .argsort (np .argsort (values ))
    return order /max (len (values )-1 ,1 )


def critical_angle_from_material (
material :np .ndarray ,q_material :np .ndarray 
)->np .ndarray :
    """Map soil texture to a per-sample critical slope in 22–38 degrees."""
    def z (column :np .ndarray )->np .ndarray :
        std =float (np .std (column ))
        return (column -float (np .mean (column )))/(std if std >1e-9 else 1.0 )

    sand =material [:,SAND_IDX [0 ]:SAND_IDX [1 ]+1 ].mean (axis =1 )
    clay =material [:,CLAY_IDX [0 ]:CLAY_IDX [1 ]+1 ].mean (axis =1 )
    cfvo =material [:,CFVO_IDX [0 ]:CFVO_IDX [1 ]+1 ].mean (axis =1 )
    awc =material [:,AWC_SLICE ].mean (axis =1 )
    index =z (sand )+0.5 *z (cfvo )-z (clay )-0.5 *z (awc )
    theta =THETA_MIN_DEG +THETA_SPAN_DEG *rank01 (index )
    return np .where (q_material >0 ,theta ,THETA_FALLBACK_DEG ).astype (np .float32 )


def boundary_strength (
terrain :np .ndarray ,valid :np .ndarray ,theta_c :float 
)->np .ndarray :
    """Sample-level boundary strength: divide + aspect break + critical-slope break."""
    plan =terrain [TERRAIN_INDEX ["plan_curvature"]].astype (np .float32 )
    slope =terrain [TERRAIN_INDEX ["slope_deg"]].astype (np .float32 )
    aspect_sin =terrain [TERRAIN_INDEX ["aspect_sin"]].astype (np .float32 )
    aspect_cos =terrain [TERRAIN_INDEX ["aspect_cos"]].astype (np .float32 )

    divide =robust_unit (np .clip (plan ,0.0 ,None ),valid )
    grad_sin =np .hypot (*np .gradient (aspect_sin ))
    grad_cos =np .hypot (*np .gradient (aspect_cos ))
    aspect_break =robust_unit (np .hypot (grad_sin ,grad_cos ),valid )
    slope_break =np .exp (
    -(((slope -theta_c )/SLOPE_BREAK_WIDTH_DEG )**2 )
    ).astype (np .float32 )
    return (divide +aspect_break +slope_break )/3.0 


def absorb_small_units (labels :np .ndarray ,mask :np .ndarray ,min_area :int )->np .ndarray :
    """，。"""
    if labels .max ()<=1 :
        return labels 
    sizes =np .bincount (labels .ravel ())
    small ={i for i in range (1 ,len (sizes ))if 0 <sizes [i ]<min_area }
    if not small :
        return labels 
    keep =np .isin (labels ,[i for i in range (1 ,len (sizes ))if sizes [i ]>=min_area ])
    if not keep .any ():
        return np .where (mask ,1 ,0 ).astype (labels .dtype )
        # 
    _ ,indices =ndimage .distance_transform_edt (~keep ,return_indices =True )
    filled =labels [tuple (indices )]
    return np .where (mask &~keep ,filled ,labels )


def split_component (
local_mask :np .ndarray ,local_boundary :np .ndarray ,min_area :int 
)->np .ndarray :
    """， 1 。"""
    if int (local_mask .sum ())<2 *min_area :
        return local_mask .astype (np .int32 )
    inside =local_boundary [local_mask ]
    cut =float (np .quantile (inside ,MARKER_QUANTILE ))
    seeds =local_mask &(local_boundary <=cut )
    markers ,count =ndimage .label (seeds ,structure =np .ones ((3 ,3 ),int ))
    if count <2 :
        return local_mask .astype (np .int32 )
    labels =watershed (local_boundary ,markers ,mask =local_mask )
    labels =absorb_small_units (labels ,local_mask ,min_area )
    # 
    unique =np .unique (labels [labels >0 ])
    remap =np .zeros (int (labels .max ())+1 ,dtype =np .int32 )
    remap [unique ]=np .arange (1 ,len (unique )+1 )
    return remap [labels ]


def process_fold (
cache_dir :Path ,
fold_id :str ,
modes :dict [str ,np .ndarray ],
min_area :int ,
ring_radius :int ,
threshold_override :float |None =None ,
)->dict [str ,list [dict ]]:
    """，。 threshold_override ， 。 receipt 。"""
    receipt =json .loads (
    (cache_dir /f"{fold_id }_oof_cache_receipt.json").read_text (encoding ="utf-8")
    )
    threshold =float (receipt ["threshold"])if threshold_override is None else threshold_override 
    with np .load (cache_dir /f"{fold_id }_oof_cache.npz",allow_pickle =False )as handle :
        sample_id =[str (item )for item in handle ["sample_id"]]
        dataset_id =[str (item )for item in handle ["dataset_id"]]
        event_id =[str (item )for item in handle ["canonical_event_id"]]
        probability_all =handle ["visual_probability"]
        target_all =handle ["target"]
        valid_all =handle ["valid"]
        terrain_all =handle ["terrain"]
    with np .load (cache_dir /f"{fold_id }_optical_cache.npz",allow_pickle =False )as handle :
        pre_all =handle ["optical_pre"]
        post_all =handle ["optical_post"]

    structure =ndimage .generate_binary_structure (2 ,2 )
    out :dict [str ,list [dict ]]={mode :[]for mode in modes }

    for index in range (len (sample_id )):
        keep =valid_all [index ].astype (bool )
        truth =target_all [index ].astype (bool )&keep 
        probability =probability_all [index ].astype (np .float32 )
        predicted =(probability >=threshold )&keep 
        if not predicted .any ():
            continue 
        component_labels ,count =ndimage .label (predicted ,structure =structure )
        if count ==0 :
            continue 
        terrain =terrain_all [index ].astype (np .float32 )
        pre_cube =pre_all [index ].astype (np .float32 )
        post_cube =post_all [index ].astype (np .float32 )
        pre_idx =spectral_index_stack (pre_cube )
        post_idx =spectral_index_stack (post_cube )
        windows =ndimage .find_objects (component_labels )

        boundary_cache :dict [float ,np .ndarray ]={}
        for mode ,theta_all in modes .items ():
            if mode =="whole":
                continue 
            theta =float (theta_all [0 ])if theta_all .ndim ==0 else float (theta_all [index ])
            if theta not in boundary_cache :
                boundary_cache [theta ]=boundary_strength (terrain ,keep ,theta )

        for label_value in range (1 ,count +1 ):
            window =windows [label_value -1 ]
            local =component_labels [window ]==label_value 
            if int (local .sum ())<min_area :
                continue 
            for mode ,theta_all in modes .items ():
                if mode =="whole":
                    units =local .astype (np .int32 )
                else :
                    theta =(
                    float (theta_all [0 ])if theta_all .ndim ==0 else float (theta_all [index ])
                    )
                    units =split_component (
                    local ,boundary_cache [theta ][window ],min_area 
                    )
                for unit_value in range (1 ,int (units .max ())+1 ):
                    unit =units ==unit_value 
                    area =int (unit .sum ())
                    if area <min_area :
                        continue 
                    rows_local ,cols_local =np .nonzero (unit )
                    rows =rows_local +window [0 ].start 
                    cols =cols_local +window [1 ].start 
                    intersection =int (np .count_nonzero (truth [rows ,cols ]))
                    row =component_features (rows ,cols ,terrain ,probability ,unit )
                    row .update (
                    component_spectral_features (
                    unit ,window ,pre_idx ,post_idx ,pre_cube ,post_cube ,
                    keep ,ring_radius ,
                    )
                    )
                    row .update (
                    {
                    "sample_id":sample_id [index ],
                    "dataset_id":dataset_id [index ],
                    "canonical_event_id":event_id [index ],
                    "component_id":int (label_value ),
                    "unit_id":int (unit_value ),
                    "parent_area_px":float (local .sum ()),
                    "purity":intersection /area ,
                    "intersection_px":float (intersection ),
                    "false_px":float (area -intersection ),
                    }
                    )
                    out [mode ].append (row )
    return out 


def main ()->None :
    parser =argparse .ArgumentParser ()
    parser .add_argument ("--cache",type =Path ,default =DEFAULT_CACHE )
    parser .add_argument ("--min-area",type =int ,default =MIN_AREA )
    parser .add_argument ("--ring-radius",type =int ,default =5 )
    parser .add_argument ("--seed",type =int ,default =20260725 )
    parser .add_argument (
    "--modes",nargs ="+",default =["whole","geomorphic","material","material_shuffled"],
    help ="text",
    )
    parser .add_argument (
    "--threshold-override",type =float ,default =None ,
    help ="text",
    )
    parser .add_argument (
    "--outdir",type =Path ,
    default =PROJECT_ROOT /"experiments/revision2026/pild_subobject_units_v1",
    )
    args =parser .parse_args ()
    started =time .time ()

    # 
    materials ,q_materials ,sample_ids ,events ,datasets =[],[],[],[],[]
    for fold_id in FOLD_IDS :
        with np .load (args .cache /f"{fold_id }_optical_cache.npz",allow_pickle =False )as handle :
            materials .append (handle ["material_features"])
            q_materials .append (handle ["q_material"])
            sample_ids .extend (str (item )for item in handle ["sample_id"])
        with np .load (args .cache /f"{fold_id }_oof_cache.npz",allow_pickle =False )as handle :
            events .extend (str (item )for item in handle ["canonical_event_id"])
            datasets .extend (str (item )for item in handle ["dataset_id"])
    material =np .concatenate (materials ).astype (np .float64 )
    q_material =np .concatenate (q_materials ).astype (np .float64 )
    theta =critical_angle_from_material (material ,q_material )

    # 
    rng =np .random .default_rng (args .seed +31 )
    events_arr =np .asarray (events )
    datasets_arr =np .asarray (datasets )
    donor =np .arange (len (material ))
    for ds in np .unique (datasets_arr ):
        rows =np .where (datasets_arr ==ds )[0 ]
        uniq =np .unique (events_arr [rows ])
        if len (uniq )<2 :
            continue 
        mapping =dict (zip (uniq ,rng .permutation (uniq )))
        for e in uniq :
            target_rows =np .where ((events_arr ==e )&(datasets_arr ==ds ))[0 ]
            pool =np .where (events_arr ==mapping [e ])[0 ]
            donor [target_rows ]=pool [rng .integers (0 ,len (pool ),size =len (target_rows ))]
    theta_shuffled =theta [donor ]

    print (
    f"： {np .median (theta ):.1f}°  "
    f"IQR [{np .percentile (theta ,25 ):.1f}°, {np .percentile (theta ,75 ):.1f}°]  "
    f" Material  {float ((q_material >0 ).mean ()):.1%}"
    )

    offset =0 
    collected :dict [str ,list [pd .DataFrame ]]={}
    for fold_id in FOLD_IDS :
        with np .load (args .cache /f"{fold_id }_optical_cache.npz",allow_pickle =False )as handle :
            n =len (handle ["sample_id"])
        available ={
        "whole":np .asarray (THETA_FALLBACK_DEG ),
        "geomorphic":np .full (n ,THETA_FALLBACK_DEG ,dtype =np .float32 ),
        "material":theta [offset :offset +n ],
        "material_shuffled":theta_shuffled [offset :offset +n ],
        }
        modes ={name :available [name ]for name in args .modes }
        offset +=n 
        result =process_fold (
        args .cache ,fold_id ,modes ,args .min_area ,args .ring_radius ,
        threshold_override =args .threshold_override ,
        )
        for mode ,rows in result .items ():
            collected .setdefault (mode ,[]).append (pd .DataFrame (rows ))
        print (
        f"{fold_id }: "
        +"  ".join (f"{mode }={len (rows ):,}"for mode ,rows in result .items ())
        )

    args .outdir .mkdir (parents =True ,exist_ok =True )
    stats ={}
    for mode ,frames in collected .items ():
        table =pd .concat (frames ,ignore_index =True )
        table .to_parquet (args .outdir /f"units_{mode }.parquet",index =False )
        mixed =(table .purity >0.05 )&(table .purity <0.60 )
        stats [mode ]={
        "n_units":int (len (table )),
        "median_area":float (table .area_px .median ()),
        "mixed_fraction":float (mixed .mean ()),
        "mixed_tp_share":float (
        table .intersection_px [mixed ].sum ()/table .intersection_px .sum ()
        ),
        "mixed_fp_share":float (table .false_px [mixed ].sum ()/table .false_px .sum ()),
        "tp_total":float (table .intersection_px .sum ()),
        "fp_total":float (table .false_px .sum ()),
        }
        print (
        f"{mode :18s}  {len (table ):7,}   {table .area_px .median ():6.1f}  "
        f" {mixed .mean ():.1%}   TP {stats [mode ]['mixed_tp_share']:.1%} / "
        f"FP {stats [mode ]['mixed_fp_share']:.1%}"
        )

    (args .outdir /"receipt.json").write_text (
    json .dumps (
    {
    "schema_version":"pild_subobject_units.v1",
    "threshold_override":args .threshold_override ,
    "min_area":args .min_area ,
    "ring_radius":args .ring_radius ,
    "marker_quantile":MARKER_QUANTILE ,
    "theta_range_deg":[THETA_MIN_DEG ,THETA_MIN_DEG +THETA_SPAN_DEG ],
    "theta_fallback_deg":THETA_FALLBACK_DEG ,
    "slope_break_width_deg":SLOPE_BREAK_WIDTH_DEG ,
    "stats":stats ,
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
