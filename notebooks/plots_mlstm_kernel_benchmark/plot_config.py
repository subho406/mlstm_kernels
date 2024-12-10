#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

kernel_colors = {
    "torch_flashattn": "#165b89",
    "torch_cudnn": "#439ebd",
    "flashattn3": '#80a8b3',
    "mamba2_noconv": "#d08814",
    "mamba2": "#d08814",
    "mamba": "#ffd449",
    "chunk_simple_gla": "#4fb72e",
    "fused_chunk_gla": "#548c2f",
    "chunk_gla": "#548c2f",
    "torch_native": "#e52e66",
    "triton_xl_chunk": "#9a3c73",
    "triton_limit_chunk": "#e52e66",
}


kernel_labels = {
    "torch_flashattn": "Torch FlashAttn",
    "torch_cudnn": "cuDNN FlashAttn",
    "flashattn3": "FlashAttn 3",
    "mamba2_noconv": "Mamba 2 SSD",
    "mamba2": "Mamba 2",
    "mamba": "Mamba",
    "chunk_simple_gla": "Simple GLA",
    "fused_chunk_gla": "GLA (fused)",
    "chunk_gla": "GLA",
    "torch_native": "mLSTM (torch)",
    "triton_xl_chunk": "mLSTM (triton XL)",
    "triton_limit_chunk": "mLSTM (triton)",
}

style_dict = {
    # FWBW
    "chunkwise--triton_limit_chunk__bfloat16__fwbw__cs-64_nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['triton_limit_chunk'],
        "label": kernel_labels['triton_limit_chunk'],
    },
    "chunkwise--triton_xl_chunk__bfloat16__fwbw__cs-128_nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['triton_xl_chunk'],
        "label": kernel_labels['triton_xl_chunk'],
    },
    "chunkwise--native_custbw____bfloat16__fwbw__cs-128_nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['torch_native'],
        "label": kernel_labels['torch_native'],
    },
    "chunkwise--native_custbw____bfloat16__fwbw__cs-256_nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['torch_native'],
        "label": kernel_labels['torch_native'],
    },
    "chunkwise--native_custbw____bfloat16__fwbw__cs-64_nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['torch_native'],
        "label": kernel_labels['torch_native'],
    },
    "chunkwise--native_autograd____bfloat16__fwbw__cs-256_nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['torch_native'],
        "label": kernel_labels['torch_native'],
    },
    "chunk_gla____bfloat16__fwbw__nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['chunk_gla'],
        "label": kernel_labels['chunk_gla'],
    },
    "fused_chunk_gla__bfloat16__fwbw__nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['fused_chunk_gla'],
        "label": kernel_labels['fused_chunk_gla'],
    },
    "chunk_simple_gla__bfloat16__fwbw__nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['chunk_simple_gla'],
        "label": kernel_labels['chunk_simple_gla'],
    },
    "mamba____bfloat16__fwbw__nh-1_hdv-8192_hdq-16": {
        "color": kernel_colors['mamba'],
        "label": kernel_labels['mamba'],
    },
    "mamba2____bfloat16__fwbw__nh-128_hdv-64_hdq-64": {
        "color": kernel_colors['mamba2'],
        "label": kernel_labels['mamba2'],
    },
    "mamba2_noconv____bfloat16__fwbw__nh-128_hdv-64_hdq-64": {
        "color": kernel_colors['mamba2_noconv'],
        "label": kernel_labels['mamba2_noconv'],
    },
    "flashattn3____bfloat16__fwbw__nh-32_hdv-128_hdq-128": {
        "color": kernel_colors['flashattn3'],
        "label": kernel_labels['flashattn3'],
    },
    "torch_flash__bfloat16__fwbw__nh-32_hdq-128_hdv-128": {
        "color": kernel_colors['torch_flashattn'],
        "label": kernel_labels['torch_flashattn'],
    },
    "torch_cudnn__bfloat16__fwbw__nh-32_hdq-128_hdv-128": {
        "color": kernel_colors['torch_cudnn'],
        "label": kernel_labels['torch_cudnn'],
    },
    # FW
    "chunkwise--triton_limit_chunk__bfloat16__fw__cs-64_nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['triton_limit_chunk'],
        "label": kernel_labels['triton_limit_chunk'],
    },
    "chunkwise--triton_xl_chunk__bfloat16__fw__cs-128_nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['triton_xl_chunk'],
        "label": kernel_labels['triton_xl_chunk'],
    },
    "chunkwise--native_custbw____bfloat16__fw__cs-128_nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['torch_native'],
        "label": kernel_labels['torch_native'],
    },
    "chunkwise--native_custbw____bfloat16__fw__cs-256_nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['torch_native'],
        "label": kernel_labels['torch_native'],
    },
    "chunkwise--native_custbw____bfloat16__fw__cs-64_nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['torch_native'],
        "label": kernel_labels['torch_native'],
    },
    "chunkwise--native_autograd____bfloat16__fw__cs-256_nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['torch_native'],
        "label": kernel_labels['torch_native'],
    },
    "chunk_gla____bfloat16__fw__nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['chunk_gla'],
        "label": kernel_labels['chunk_gla'],
    },
    "fused_chunk_gla__bfloat16__fw__nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['fused_chunk_gla'],
        "label": kernel_labels['fused_chunk_gla'],
    },
    "chunk_simple_gla__bfloat16__fw__nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['chunk_simple_gla'],
        "label": kernel_labels['chunk_simple_gla'],
    },
    "mamba____bfloat16__fw__nh-1_hdv-8192_hdq-16": {
        "color": kernel_colors['mamba'],
        "label": kernel_labels['mamba'],
    },
    "mamba2____bfloat16__fw__nh-128_hdv-64_hdq-64": {
        "color": kernel_colors['mamba2'],
        "label": kernel_labels['mamba2'],
    },
    "mamba2_noconv____bfloat16__fw__nh-128_hdv-64_hdq-64": {
        "color": kernel_colors['mamba2_noconv'],
        "label": kernel_labels['mamba2_noconv'],
    },
    "flashattn3____bfloat16__fw__nh-32_hdv-128_hdq-128": {
        "color": kernel_colors['flashattn3'],
        "label": kernel_labels['flashattn3'],
    },
    "torch_flash__bfloat16__fw__nh-32_hdq-128_hdv-128": {
        "color": kernel_colors['torch_flashattn'],
        "label": kernel_labels['torch_flashattn'],
    },
    "torch_cudnn__bfloat16__fw__nh-32_hdq-128_hdv-128": {
        "color": kernel_colors['torch_cudnn'],
        "label": kernel_labels['torch_cudnn'],
    },
    # Generation
    "triton_fused__bfloat16__fw__nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['triton_xl_chunk'],
        "label": kernel_labels['triton_xl_chunk'],
    },
    "triton_fused__float32__fw__nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['triton_limit_chunk'],
        "label": kernel_labels['triton_limit_chunk'],
    },
    "native____bfloat16__fw__nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['triton_limit_chunk'],
        "label": kernel_labels['triton_limit_chunk'],
    },
    "native____float32__fw__nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['triton_limit_chunk'],
        "label": kernel_labels['triton_limit_chunk'],
    },
    "fused_recurrent_gla____bfloat16__fw__nh-8_hdv-512_hdq-256": {
        "color": kernel_colors['fused_chunk_gla'],
        "label": kernel_labels['fused_chunk_gla'],
    }
}

# col_order_fwbw = [
    
#     "chunkwise--triton_xl_chunk__bfloat16__fwbw__cs-128_nh-8_hdv-512_hdq-256",
#     "chunkwise--triton_limit_chunk__bfloat16__fwbw__cs-64_nh-8_hdv-512_hdq-256",
#     # "chunkwise--native_custbw____bfloat16__fwbw__cs-128_nh-8_hdv-512_hdq-256",
#     # "chunkwise--native_custbw____bfloat16__fwbw__cs-256_nh-8_hdv-512_hdq-256",
#     # "chunkwise--native_custbw____bfloat16__fwbw__cs-64_nh-8_hdv-512_hdq-256",
#     # "chunkwise--native_autograd____bfloat16__fwbw__cs-256_nh-8_hdv-512_hdq-256",
#     "chunk_gla____bfloat16__fwbw__nh-8_hdv-512_hdq-256",
#     # "fused_chunk_gla__bfloat16__fwbw__nh-8_hdv-512_hdq-256",
#     # "chunk_simple_gla__bfloat16__fwbw__nh-8_hdv-512_hdq-256",
#     "mamba____bfloat16__fwbw__nh-1_hdv-8192_hdq-16",
#     "mamba2____bfloat16__fwbw__nh-128_hdv-64_hdq-64",
#     # "mamba2_noconv____bfloat16__fwbw__nh-64_hdv-64_hdq-64",
#     "torch_flash__bfloat16__fwbw__nh-32_hdq-128_hdv-128",
#     "torch_cudnn__bfloat16__fwbw__nh-32_hdq-128_hdv-128",
#     "flashattn3____bfloat16__fwbw__nh-32_hdv-128_hdq-128"
# ]

# use reversed col order for line plots
col_order_fwbw = [
    "torch_flash__bfloat16__fwbw__nh-32_hdq-128_hdv-128",
    "torch_cudnn__bfloat16__fwbw__nh-32_hdq-128_hdv-128",
    "flashattn3____bfloat16__fwbw__nh-32_hdv-128_hdq-128",
    "mamba____bfloat16__fwbw__nh-1_hdv-8192_hdq-16",
    "mamba2____bfloat16__fwbw__nh-128_hdv-64_hdq-64",
    "chunk_gla____bfloat16__fwbw__nh-8_hdv-512_hdq-256",
    # "fused_chunk_gla__bfloat16__fwbw__nh-8_hdv-512_hdq-256",
    # "chunk_simple_gla__bfloat16__fwbw__nh-8_hdv-512_hdq-256",
    # "mamba2_noconv____bfloat16__fwbw__nh-64_hdv-64_hdq-64", 
    "chunkwise--triton_limit_chunk__bfloat16__fwbw__cs-64_nh-8_hdv-512_hdq-256",
    "chunkwise--triton_xl_chunk__bfloat16__fwbw__cs-128_nh-8_hdv-512_hdq-256",
    # "chunkwise--native_custbw____bfloat16__fwbw__cs-128_nh-8_hdv-512_hdq-256",
    # "chunkwise--native_custbw____bfloat16__fwbw__cs-256_nh-8_hdv-512_hdq-256",
    # "chunkwise--native_custbw____bfloat16__fwbw__cs-64_nh-8_hdv-512_hdq-256",
    # "chunkwise--native_autograd____bfloat16__fwbw__cs-256_nh-8_hdv-512_hdq-256",
]

legend_order = [
    "mLSTM (triton XL)",
    "mLSTM (triton)",
    "GLA",
    "Mamba",
    "Mamba 2",
    "Torch FlashAttn",
    "cuDNN FlashAttn",
    "FlashAttn 3",
    # "Mamba 2 SSD",
    # "Simple GLA",
    # "GLA (fused)",
    # "mLSTM (torch)",
]


# col_order_fw = [
#     "chunkwise--triton_xl_chunk__bfloat16__fw__cs-128_nh-8_hdv-512_hdq-256",
#     "chunkwise--triton_limit_chunk__bfloat16__fw__cs-64_nh-8_hdv-512_hdq-256",
#     # "chunkwise--native_custbw____bfloat16__fw__cs-128_nh-8_hdv-512_hdq-256",
#     # "chunkwise--native_custbw____bfloat16__fw__cs-256_nh-8_hdv-512_hdq-256",
#     # "chunkwise--native_custbw____bfloat16__fw__cs-64_nh-8_hdv-512_hdq-256",
#     # "chunkwise--native_autograd____bfloat16__fw__cs-256_nh-8_hdv-512_hdq-256",
#     "chunk_gla____bfloat16__fw__nh-8_hdv-512_hdq-256",
#     # "fused_chunk_gla__bfloat16__fw__nh-8_hdv-512_hdq-256",
#     # "chunk_simple_gla__bfloat16__fw__nh-8_hdv-512_hdq-256",
#     "mamba____bfloat16__fw__nh-1_hdv-8192_hdq-16",
#     "mamba2____bfloat16__fw__nh-128_hdv-64_hdq-64",
#     # "mamba2_noconv____bfloat16__fw__nh-64_hdv-64_hdq-64",
#     "torch_flash__bfloat16__fw__nh-32_hdq-128_hdv-128",
#     "torch_cudnn__bfloat16__fw__nh-32_hdq-128_hdv-128",
#     "flashattn3____bfloat16__fw__nh-32_hdv-128_hdq-128",
# ]
# use reversed col order for line plots
col_order_fw = [
    "torch_flash__bfloat16__fw__nh-32_hdq-128_hdv-128",
    "torch_cudnn__bfloat16__fw__nh-32_hdq-128_hdv-128",
    "flashattn3____bfloat16__fw__nh-32_hdv-128_hdq-128",
    "mamba____bfloat16__fw__nh-1_hdv-8192_hdq-16",
    "mamba2____bfloat16__fw__nh-128_hdv-64_hdq-64",
    "chunk_gla____bfloat16__fw__nh-8_hdv-512_hdq-256",
    # "fused_chunk_gla__bfloat16__fw__nh-8_hdv-512_hdq-256",
    # "chunk_simple_gla__bfloat16__fw__nh-8_hdv-512_hdq-256",
    # "mamba2_noconv____bfloat16__fw__nh-64_hdv-64_hdq-64",
    "chunkwise--triton_limit_chunk__bfloat16__fw__cs-64_nh-8_hdv-512_hdq-256",
    "chunkwise--triton_xl_chunk__bfloat16__fw__cs-128_nh-8_hdv-512_hdq-256",
    # "chunkwise--native_custbw____bfloat16__fw__cs-128_nh-8_hdv-512_hdq-256",
    # "chunkwise--native_custbw____bfloat16__fw__cs-256_nh-8_hdv-512_hdq-256",
    # "chunkwise--native_custbw____bfloat16__fw__cs-64_nh-8_hdv-512_hdq-256",
    # "chunkwise--native_autograd____bfloat16__fw__cs-256_nh-8_hdv-512_hdq-256",
]

col_order_gen = [
    "triton_fused__bfloat16__fw__nh-8_hdv-512_hdq-256",
    # "triton_fused__float32__fw__nh-8_hdv-512_hdq-512",
    # "native____bfloat16__fw__nh-8_hdv-512_hdq-512",
    # "native____float32__fw__nh-8_hdv-512_hdq-512",
    "mamba____bfloat16__fw__nh-1_hdv-8192_hdq-16",
    "mamba2____bfloat16__fw__nh-128_hdv-64_hdq-64",
    "fused_recurrent_gla____bfloat16__fw__nh-8_hdv-512_hdq-256",
    # "fused_recurrent_simple_gla____bfloat16__fw__nh-8_hdv-512_hdq-256",

]