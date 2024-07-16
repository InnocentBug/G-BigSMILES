# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022: Ludwig Schneider
# See LICENSE for details

import copy

import numpy as np
from rdkit import Chem

import gbigsmiles


def test_molecule():
    test_args = [
        (
            "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}CC{[>][<]CC([>])c1ccccc1[<]}C(C)CC(c1ccccc1)c1ccccc1",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>]{[>][<]CC([>])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1",
            "[H][>|0.0|]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>|0.0|]{[>][<]CC([>])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1",
            None,
            None,
            None,
        ),
        (
            "{[] CC([$])=NCC[$]; [H][$][]}|schulz_zimm(1000, 900)|",
            "{[]CC([$])=NCC[$]; [H][$][]}",
            "{[]CC([$])=NCC[$]; [H][$][]}|schulz_zimm(1000.0, 900.0)|",
            "[H]CCN=C(C)CCN=C(C)CCN=C(C)CCN=C(C)CCN=C(C)CCN=C(C)C(C)=NCCC(C)=NCCC(C)=NCCCCN=C(C)C(C)=NCCC(C)=NCCC(C)=NCCC(C)=NCCC(C)=NCCC(C)=NCCCCN=C([H])C",
            None,
            "[H]CCN=C(C)CCN=C(C)CCN=C(C)CCN=C(C)C(C)=NCCC(C)=NCCC(C)=NCCC(C)=NCCC(C)=NCCCCN=C(C)C(C)=NCCCCN=C(C)CCN=C(C)CCN=C(C)CCN=C(C)CCN=C(C)CCN=C(C)CCN=C(C)CCN=CC",
        ),
        (
            "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|schulz_zimm(1500, 1400)|[<]CC.|60000|",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC.",
            "[H][>|0.0|]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|schulz_zimm(1500.0, 1400.0)|[<]CC.|60000.0|",
            "[H]CC(C)(CC(C)(CC(C)(CC(C)(CC(C)(CC)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F",
            "[H]CC(C)(CC(C)(CC(C)(CC(C)(CC(C)(CC)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F",
            "[H]CC(C)(CC(C)(CC(C)(CC(C)(CC(C)(CC(C)(CC)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F",
        ),
        (
            "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|schulz_zimm(1500, 1000)|CC{[>][>]CC([<])c1ccccc1[<]}|schulz_zimm(1500, 1000)|C(C)CC(c1ccccc1)c1ccccc1.|60000|",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>]{[>][>]CC([<])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1.",
            "[H][>|0.0|]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|schulz_zimm(1500.0, 1000.0)|[<]CC[>|0.0|]{[>][>]CC([<])c1ccccc1[<]}|schulz_zimm(1500.0, 1000.0)|[<]C(C)CC(c1ccccc1)c1ccccc1.|60000.0|",
            "[H]CC(C)(CCC(CC(CC(CC(CC(CC(CC(CC(CC(C)CC(c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F",
            "[H]CC(C)(CC(C)(CC(C)(CC(C)(CC(C)(CCC(CC(CC(CC(C)CC(c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F",
            "[H]CC(C)(CC(C)(CC(C)(CC(C)(CC(C)(CC(C)(CCC(CC(CC(CC(CC(CC(CC(CC(CC(CC(CC(CC(CC(CC(CC(CC(CC(CC(CC(CC(C)CC(c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F",
        ),
        (
            "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}CC{[>][<]CC([>])c1ccccc1[<]}|gauss(500, 10)|C(C)CC(c1ccccc1)c1ccccc1.|60000|",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>]{[>][<]CC([>])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1.",
            "[H][>|0.0|]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>|0.0|]{[>][<]CC([>])c1ccccc1[<]}|gauss(500.0, 10.0)|[<]C(C)CC(c1ccccc1)c1ccccc1.|60000.0|",
            None,
            None,
            None,
        ),
        (
            "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|schulz_zimm(1500, 1000)|CC{[>][<]CC([>])c1ccccc1[<]}|schulz_zimm(1500, 1000)|C(C)CC(c1ccccc1)c1ccccc1.|50%|",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}[<]CC[>]{[>][<]CC([>])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1.",
            "[H][>|0.0|]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F[<]}|schulz_zimm(1500.0, 1000.0)|[<]CC[>|0.0|]{[>][<]CC([>])c1ccccc1[<]}|schulz_zimm(1500.0, 1000.0)|[<]C(C)CC(c1ccccc1)c1ccccc1.|50.0%|",
            "[H]CC(C)(CCCC(CC(CC(CC(CC(CC(CC(CC(c1ccccc1)C(C)CC(c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F",
            "[H]CC(C)(CC(C)(CC(C)(CC(C)(CC(C)(CCCC(CC(CC(c1ccccc1)C(C)CC(c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F",
            "[H]CC(C)(CC(C)(CC(C)(CC(C)(CC(C)(CC(C)(CCCC(CC(CC(CC(CC(CC(CC(CC(CC(CC(CC(CC(CC(CC(CC(CC(CC(CC(CC(c1ccccc1)C(C)CC(c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F)C(=O)OCC(O)CSc1c(F)c(F)c(F)c(F)c1F",
        ),
        (
            "[H]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F[<]}|schulz_zimm(1000, 950)|CC{[>][<]CC([>])c1ccccc1[<]}|schulz_zimm(500, 400)|C(C)CC(c1ccccc1)c1ccccc1.|5e7|",
            "[H][>]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F[<]}[<]CC[>]{[>][<]CC([>])c1ccccc1[<]}[<]C(C)CC(c1ccccc1)c1ccccc1.",
            "[H][>|0.0|]{[>]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F[<]}|schulz_zimm(1000.0, 950.0)|[<]CC[>|0.0|]{[>][<]CC([>])c1ccccc1[<]}|schulz_zimm(500.0, 400.0)|[<]C(C)CC(c1ccccc1)c1ccccc1.|50000000.0|",
            "[H]CC(C)(CC(C)(CC(C)(CC(C)(CC(C)(CCCC(CC(CC(CC(c1ccccc1)C(C)CC(c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSc1ccc(F)c(F)c1)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSc1ccc(F)c(F)c1",
            "[H]CC(C)(CC(C)(CC(C)(CC(C)(CC(C)(CCCC(CC(CC(CC(CC(CC(c1ccccc1)C(C)CC(c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)C(=O)OCC(O)CSc1ccc(F)c(F)c1)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSc1ccc(F)c(F)c1)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSc1ccc(F)c(F)c1",
            "[H]CC(C)(CC(C)(CC(C)(CC(C)(CC(C)(CCCC(CC(CC(CC(CC(c1ccccc1)C(C)CC(c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)C(=O)OCC(O)CSc1ccc(F)c(F)c1)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSc1ccc(F)c(F)c1)C(=O)OCC(O)CSC(F)(F)F)C(=O)OCC(O)CSc1ccc(F)c(F)c1",
        ),
        (
            "{[]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][N][<]}|schulz_zimm(1000, 450)|{[>][<]CC([>])c1ccccc1; [>]N, [<][H][]}|schulz_zimm(400, 300)|.|5e7|",
            "{[]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][N][<]}{[>][<]CC([>])c1ccccc1; [>]N, [<][H][]}.",
            "{[]CC([>])(C[<])C(=O)OCC(O)CSc1ccc(F)c(F)c1, CC([>])(C[<])C(=O)OCC(O)CSC(F)(F)F; [>][N][<]}|schulz_zimm(1000.0, 450.0)|{[>][<]CC([>])c1ccccc1; [>]N, [<][H][]}|schulz_zimm(400.0, 300.0)|.|50000000.0|",
            "[H]C(CC(CC(CC(CC(CC(CC(CC(CC(C)(C[N])C(=O)OCC(O)CSC(F)(F)F)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1",
            "[H]C(CC(CC(C)(C[N])C(=O)OCC(O)CSc1ccc(F)c(F)c1)c1ccccc1)c1ccccc1",
            "CC(C)(CC(CC(CC(CCc1ccccc1)c1ccccc1)c1ccccc1)c1ccccc1)C(=O)OCC(O)CSc1ccc(F)c(F)c1",
        ),
        (
            "OC{[>] [<]CC[>], [<|.5|]C(N[>|.1 0 0 0 0 0 0|])C[>]; [<][H], [<]C [<]}|schulz_zimm(5000, 4500)|COOC{[<] [<]COC[>], [<]C(ON)C[>] [>]}|schulz_zimm(5000, 4500)|{[<] [<]COCOC[>], [<]CONOC[>] [>]}|schulz_zimm(1700, 1500)|F",
            "OC[>]{[>][<]CC[>], [<]C(N[>])C[>]; [<][H], [<]C[<]}[<]COOC[<]{[<][<]COC[>], [<]C(ON)C[>][>]}{[<][<]COCOC[>], [<]CONOC[>][>]}[>]F",
            "OC[>|0.0|]{[>][<]CC[>], [<|0.5|]C(N[>|0.1 0.0 0.0 0.0 0.0 0.0 0.0|])C[>]; [<][H], [<]C[<]}|schulz_zimm(5000.0, 4500.0)|[<]COOC[<|0.0|]{[<][<]COC[>], [<]C(ON)C[>][>]}|schulz_zimm(5000.0, 4500.0)|{[<][<]COCOC[>], [<]CONOC[>][>]}|schulz_zimm(1700.0, 1500.0)|[>]F",
            "[H]CCCCCC(CCCC(CCCC(CCCC(CCCC(CCNC(CCCCCC(CCCC(CC(CCCCCO)NCCCCCCC(CCCC(CCCC(CCCCCC)N[H])N[H])N[H])NCCCCCCC(CCCCCC(CCCCCC(CCCCCC(CCCCCCCC(CCCC(CCCC)NC)NCCC(CCCC)NC)NC)NC)N[H])NCCC(CC(CC(C[H])NC)N[H])N[H])NCCCCCCCCCCC(CC(CCCC)NC)NC)CC(CC(CC(CCCCCCCCCCCCCC(CCCCCCCCCCCCCCCCCCCCCCCC)NCCCCCCC(CC)N[H])NCCCCC(CCCC(CCCCCCCC(CC(CCCCCC)N[H])N[H])NC)NCCCCC)NCCCCC(CCCC(CC(CCCCCC(CCCCCCCC)NC)N[H])N[H])NCOOCCC(COCCOCCC(COCCOCCOCCC(COCCOCCC(COCCC(COCCOCCOCCC(COCCC(COCCC(COCCOCCOCCOCCOCCC(CC(CC(COCCOCCOCCOCCC(CC(CC(COCCOCCC(CC(CC(CC(CC(COCCC(CC(CC(CC(CC(COCCOCCOCCC(CC(COCCOCCOCCC(CC(CC(COCCOCCOCCC(COCCOCCOCCOCCOCCC(COCCC(CC(COCCOCCOCCOCCOCCOCCC(CC(COCCOCCOCCOCCC(COCCOCCC(CC(CC(CC(CC(COCCC(COCCOCCC(COCCC(CC(CONOCCOCOCCOCOCCOCOCCOCOCCOCOCCONOCCONOCCOCOCCONOCCONOCF)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)NCCC(CCCCCCCCCCCCCCCC(CCCC)NCC[H])NCC[H])N[H])NCCCCCCCCCCC(CCCC(CC(CCCC(CC)NC)N[H])NCC[H])N[H])NCCCCC)N[H])NC",
            "[H]CCCCCCCCCCCCCCCCCCCC(CCNC(CCCCCCCCCCCC(CCCC(CCCCCC(CC(CC(CC)NC)N[H])NC)N[H])N[H])CCCCCCCCCC(CCCC(CC(CC(NCCC(CCCC(CCCCCC(CCCC(CC(CCCCCCCCCC)NCCC(C[H])NC)NCCCCC)NC)NCCC(CCCCCCCC(CCCCCCCC(CC)N[H])N[H])NCC[H])NC)NC(CCCC(CCCCCCCCCCCC(CC(CCCC(CCCC(CC(CCCCCCCCCC(CCCC(CC(CC(CCCCCCC[H])N[H])NC)NCCCCC(CC)N[H])NCCC(CCCC(CC)NCCC(CCCC)N[H])N[H])N[H])NCC[H])NCCC(CC(CCCCCCCC)NC)NC)NCCC(CCCCCCCC)NC)NCC[H])NCCCCC(CCCCCCCCCCCCCC(CC(CC)N[H])NC)NCC[H])CC(CC(CCCCCCCCCC(CCCC(CCCC(CCCCCC(N[H])NC(CO)CCCCCCCCC[H])NCCCCCC[H])N[H])NCCCCCCCCCCCCCCC(CC(CCCCCC)NCCC(CCCCCCCCC[H])N[H])N[H])NCCCCCCCCCCCCCCC(CC(CC)N[H])N[H])NCCCCC(CC(CCCC(CCCC(CC(CC)N[H])N[H])NC)NCCC)NCC[H])NCCC(CC(CCCC(CCCC(CC)NC)N[H])N[H])NC)NCOOCCOCCOCCOCCC(CC(COCCC(COCCC(CC(CC(COCCC(CC(COCCC(CC(COCCC(COCCC(CC(CC(COCCC(COCCOCCC(CC(CC(COCCOCCC(COCCC(COCCOCCC(CC(CC(CC(CC(COCCC(COCCOCCC(CC(CC(COCCOCCOCCC(COCCC(CC(CC(COCCOCCC(COCCC(COCCC(COCCOCCOCCC(COCCC(COCCC(CC(CC(CC(CC(COCCC(COCCOCCC(COCCOCCOCCOCOCCONOCCOCOCCOCOCCOCOCCOCOCCOCOCCOCOCCONOCCONOCCONOCCOCOCCOCOCCOCOCCONOCCOCOCCONOCCONOCCOCOCCOCOCF)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)NC)N[H]",
            "[H]CCCCCCCCCCCCCCCCCC(CCCCCCCCCCCCCC(CCCCCCCCCC(CC(CC(CC(CC(CC(CCCCCC(CC(CCCCCO)NCCC(CCCCCCCCCCCC(CCCC(CCCCCC)NC)NCCC(CCCCCCCC(CC(C[H])NC)NC)NC)NCCCCCCCC[H])NCCC(CCCCCCCCCCCCCCCCCCCC)NC)NCCC(CCCCCCCC(CCCCCCCCCCCCCCCCCCCC(CCCCCCCCCC(CCCCCC(C[H])NC)NC)NC)NCCCCCCCCC(CC)NCCCCCCC)NCCC(CC(CCCCCCCC(CC(CCCCCCCCC[H])NC)NC)NC)NC)NCCCCCCC(CCCCCCCC)NC)NCCCCCCCCC(CC(CC(CCCCC[H])NCCCCCCCC[H])NC)NC)NCCCCCCCCCCCCCCCCCCC(CCCC(CCCC(CCCCCC(CC(CCCCCC)NC)NC)NC)NC)NCCC(CC(CCCC(CCCC(CCCC(CC(CCCC(CC)NC)NCCCCC)NC)NC)NCCC)NCC[H])NC)NCCCCCCCCCCC(CC(CC)NC)NC)NCCCCCCC(CCCCCCCC(CCCCCC(CCCCCC(CC(CCCCCC)NC)NCCC)NC)NC)NC)NCCCCCCC(CC(CC(C[H])NC)NCOOCCC(COCCOCCC(COCCOCCOCCC(COCCOCCOCCC(CC(COCCC(COCCC(CC(COCCC(CC(CC(CC(CC(CC(CC(CC(CC(CC(COCCC(COCCOCCC(COCCC(CC(COCCC(COCCC(CC(COCCOCCOCCC(CC(COCCOCCOCCC(CC(CC(COCCOCCC(CC(COCCOCCOCCC(COCCOCCC(CC(COCCOCCC(COCCOCCOCCC(CC(CC(COCCC(COCCOCCOCCOCCOCCC(COCCC(CC(COCCOCCOCCC(CC(CC(COCCC(COCCC(CC(CC(COCCC(CC(CC(CC(COCCOCCOCCC(CC(CC(COCCOCCC(CC(CC(COCCOCCC(CC(CC(COCCC(COCCOCCOCCOCCOCCOCCC(COCCOCCOCCC(CONOCCONOCCOCOCCONOCCOCOCCOCOCCOCOCCOCOCCONOCCONOCCOCOCCONOCCONOCCOCOCCONOCCOCOCCOCOCCONOCCOCOCF)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)NC)NCCCCC(CCCC)NC",
        ),
        (
            "OC{[>] [<]CC[>], [<|.5|]C(N[>|.1 0 0 0 0 0 0|])C[>]; [<][H], [<]C [<]}|schulz_zimm(5000, 4000)|COOC{[<] [<]COC[>], [<]C(ON)C[>] [>]}|schulz_zimm(5000, 4500)|{[<] [<]COCOC[>], [<]CONOC[>] [>]}|schulz_zimm(1700, 1600)|F",
            "OC[>]{[>][<]CC[>], [<]C(N[>])C[>]; [<][H], [<]C[<]}[<]COOC[<]{[<][<]COC[>], [<]C(ON)C[>][>]}{[<][<]COCOC[>], [<]CONOC[>][>]}[>]F",
            "OC[>|0.0|]{[>][<]CC[>], [<|0.5|]C(N[>|0.1 0.0 0.0 0.0 0.0 0.0 0.0|])C[>]; [<][H], [<]C[<]}|schulz_zimm(5000.0, 4000.0)|[<]COOC[<|0.0|]{[<][<]COC[>], [<]C(ON)C[>][>]}|schulz_zimm(5000.0, 4500.0)|{[<][<]COCOC[>], [<]CONOC[>][>]}|schulz_zimm(1700.0, 1600.0)|[>]F",
            "[H]CCCCCCCC(CCCCCCCCCCCCCC(CC(CC(CCNC(CCCCCCCCCCCCCCCCCO)CCCCCCCCCCCCCC(CC(CCCC(CCCC(CCCCCC(CCCCCC(CCCCCC(C[H])NC)N[H])N[H])NC)NCCC(CCCC)NC)NCC[H])NCCCCCCCOOCCC(CC(COCCC(COCCOCCOCCC(COCCOCCOCCC(COCCOCCC(CC(CC(CC(CC(CC(COCCOCCC(COCCOCCOCCOCCOCCOCCC(COCCC(CC(CC(CC(COCCOCCC(CC(CC(COCCOCCOCCC(CC(CC(CC(COCCC(COCCC(COCCOCCOCCOCCOCCOCCOCCOCCC(COCCOCCC(COCCC(COCCC(CC(CC(CC(CC(CC(CC(CC(CC(CC(COCCOCCOCCOCCOCCOCCC(CC(CC(CC(CC(CONOCCONOCCONOCCONOCCOCOCCONOCCONOCCONOCCONOCCONOCCOCOCCONOCCONOCCONOCCOCOCCOCOCCOCOCCOCOCCOCOCCOCOCCOCOCCOCOCCOCOCCOCOCCOCOCCOCOCCONOCF)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)NC)NCCCCC(CCC[H])N[H])NC)NC",
            "[H]CCCCCCCCCCCCNC(CCNC(CCOOCCC(COCCC(COCCOCCC(CC(COCCC(CC(COCCC(COCCC(CC(CC(CC(COCCC(CC(CC(COCCC(CC(COCCOCCC(COCCC(CC(CC(COCCC(COCCOCCOCCC(CC(CC(COCCC(COCCC(CC(CC(CC(CC(COCCOCCOCCC(COCCOCCONOCCOCOCCOCOCCOCOCCOCOCCOCOCCOCOCCONOCCOCOCCOCOCCOCOCCOCOCCONOCCOCOCCOCOCCOCOCF)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)CC(CCCCCC(CCCC(CO)NC)N[H])N[H])CC(CC(CCCC(CCCCCCCCCCCCCC(CCCCCC(CC(CC(CC(CC)N[H])NC)NCCC)N[H])NCCC(CCCCC[H])N[H])NCCC(CCC[H])NCCC)NC)NCCCCCCC(CCCCCCCC(C[H])NC)N[H]",
            "[H]CCCCCCCCCCCCCCCCCC(CCCCCCCCCCCCCC(CCCCCCCCCC(CC(CC(CC(CC(CC(CCCCCC(CC(CCCCCO)NCCC(CCCCCCCCCCCC(CCCC(CCCCCC)NC)NCCC(CCCCCCCC(CC(C[H])NC)NC)NC)NCCCCCCCC[H])NCCC(CCCCCCCCCCCCCCCCCCCC)NC)NCCC(CCCCCCCC(CCCCCCCCCCCCCCCCCCCC(CCCCCCCCCC(CCCCCC(C[H])NC)NC)NC)NCCCCCCCCC(CC)NCCCCCCC)NCCC(CC(CCCCCCCC(CC(CCCCCCCCC[H])NC)NC)NC)NC)NCCCCCCC(CCCCCCCC)NC)NCCCCCCCCC(CC(CC(CCCCC[H])NCCCCCCCC[H])NC)NC)NCCCCCCCCCCCCCCCCCCC(CCCC(CCCC(CCCCCC(CC(CCCCCC)NC)NC)NC)NC)NCCC(CC(CCCC(CCCC(CCCC(CC(CCCC(CC)NC)NCCCCC)NC)NC)NCCC)NCC[H])NC)NCCCCCCCCCCC(CC(CC)NC)NC)NCCCCCCC(CCCCCCCC(CCCCCC(CCCCCC(CC(CCCCCC)NC)NCCC)NC)NC)NC)NCCCCCCC(CC(CC(C[H])NC)NCOOCCC(CC(COCCC(CC(COCCC(COCCC(CC(COCCOCCOCCC(COCCC(CC(CC(CC(COCCOCCC(CC(COCCC(CC(CC(CC(CC(CC(COCCC(CC(CC(COCCC(COCCC(COCCOCCOCCC(COCCOCCOCCC(COCCC(COCCOCCOCCC(COCCOCCC(COCCC(COCCOCCOCCC(COCCOCCC(COCCOCCC(CC(COCCC(CC(CC(CC(CC(CC(CC(COCCC(CC(CC(COCCC(CC(CC(CC(CC(COCCOCCC(CC(COCCC(CC(COCCOCCOCCC(COCCOCCOCCC(COCCC(CC(CC(CC(CC(COCCC(CC(COCCOCCC(CC(CC(COCOCCONOCCONOCCOCOCCONOCCONOCCONOCCOCOCCOCOCCONOCCOCOCCONOCCOCOCCONOCCONOCCONOCCOCOCCOCOCCONOCCONOCF)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)ON)NC)NCCCCC(CCCC)NC",
        ),
    ]

    global_rng = np.random.Generator(np.random.MT19937(42))
    test = global_rng.uniform()
    assert test == 0.5419938930062744

    for text, big, ref, gen, mir_gen, graph_gen in test_args:
        rng = copy.deepcopy(global_rng)
        mol = gbigsmiles.Molecule(text)
        assert str(mol) == ref
        assert mol.generate_string(False) == big
        assert mol.generable == (gen is not None)

        mol_mirror = mol.gen_mirror()

        schulz_zimm_distribution = True
        # Check distribution possibility
        for element in mol.elements:
            if isinstance(element, gbigsmiles.stochastic.Stochastic):
                if not isinstance(element.distribution, gbigsmiles.distribution.SchulzZimm):
                    schulz_zimm_distribution = False
                    break

        stochastic_graph = mol.gen_stochastic_atom_graph(schulz_zimm_distribution)
        if schulz_zimm_distribution:
            full_atom_graph = gbigsmiles.AtomGraph(stochastic_graph, rng=rng)
            full_atom_graph.generate()
            graph_smi = Chem.MolToSmiles(full_atom_graph.to_mol())
            assert graph_smi == graph_gen

        if mol.generable:
            gen_mol = mol.generate(rng=copy.deepcopy(rng))
            assert gen == gen_mol.smiles

            if mir_gen:
                mirror_gen = mol_mirror.generate(rng=copy.deepcopy(rng)).smiles
                assert mirror_gen == mir_gen
            mol.gen_reaction_graph()


if __name__ == "__main__":
    test_molecule()
