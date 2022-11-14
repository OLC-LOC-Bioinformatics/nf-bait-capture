#!/usr/bin/env python

"""
Parse bait-capture read mapping results by target clusters (ARG, replicon group)
, and output a table of counts for the most abundant target per cluster.

Original script by Julie Shay and rewritten by Liam Brown.

Dependencies: pandas=1.5.1, numpy=1.23.4
Other package versions may work but are untested.
"""

__author__ = 'Liam Brown'
__copyright__ = 'Crown Copyright 2022'
__license__ = 'GPL3'
__email__ = 'liam.brown@inspection.gc.ca'
__status__ = 'Prototype'

import os
import sys
import argparse
import pandas as pd
import numpy as np

#-------------------------------------------------------------------------------
# Custom exceptions
#-------------------------------------------------------------------------------

class HeaderFormatError(Exception):
    """
    Raise this error when the values in the header of the provided tab-separated
    file do not match the expected values.
    """
    pass

#-------------------------------------------------------------------------------
# parse_arguments()
#-------------------------------------------------------------------------------

def parse_arguments():
    """
    Parse command-line arguments.

    :returns args: List of parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description = """
        Parse bait-capture read mapping results by target clusters (ARG,
        replicon group), and output a table of counts for the most abundant
        target per cluster.        
        """)

    # Required arguments
    required_args = parser.add_argument_group('Required')
    required_args.add_argument('-i', '--input', type = str, required = True,
        help = """
        Path to a tab-separated table of alignment results produced by running 
        `readcounts.py` on a sorted BAM file and the FASTA file of the bait-
        capture target database. Columns should be 'Gene', 'Hits', and 'Gene
        Fraction'.
        """)
    required_args.add_argument('-a', '--arg_clusters', type = str, 
        help = """
        Path to tab-separated table of ARG clusters for determining the most
        abundant gene per cluster. Columns should be 'ID', 'Cluster', 'Class',
        and 'Gene'.
        """)
    required_args.add_argument('-p', '--plasmid_clusters', type = str, 
        help = """
        Path to tab-separated table of plasmid replicon group clusters for
        determining the most abundant replicon group per cluster. Columns should
        be 'ID', 'Cluster', 'Replicon', and 'Replicon_group'.
        """)

    # Optional arguments
    optional_args = parser.add_argument_group('Optional')

    optional_args.add_argument('-o', '--output', type = str,
        default = 'cluster_results.tsv',
        help = """
        Path for an output tab-separated table of read mapping results parsed by
        target clusters.
        Default: 'cluster_results.tsv'
        """)
    optional_args.add_argument('-m', '--min_prop', type = float,
        default = 0,
        help = """
        The minimum proportion (coverage) of the target's total length that must
        be covered by aligned reads to be considered for the clustering step.
        Targets with mapped reads that cover less than this proportion of the
        target's total length will be filtered out.
        Default: '0'
        """)
    optional_args.add_argument('-f', '--flagstat', type = str,
        help = """
        Path to file produced by running `samtools flagstat` on a
        sorted BAM file of aligned reads to the bait-capture target database.        
        """)

    # If no arguments provided:
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

#-------------------------------------------------------------------------------
# Other functions
#-------------------------------------------------------------------------------

def read_cluster_files(arg_clusters, plasmid_clusters):
    """
    Read the cluster files provided to --arg_clusters and --plasmid_clusters and
    check for the expected formatting.

    :param arg_clusters: Path to tab-separated table of ARG clusters for 
    determining the most abundant gene per cluster.
    :type arg_clusters: str
    :param plasmid_clusters: Path to tab-separated table of replicon group 
    clusters for determining the most abundant replicon group per cluster.
    :type plasmid_clusters: str
    :returns arg_clusters_df: ARG clusters.
    :rtype arg_clusters_df: <class 'pandas.core.frame.DataFrame'>
    :returns arg_clusters_df: Replicon group clusters.
    :rtype arg_clusters_df: <class 'pandas.core.frame.DataFrame'>
    """

    arg_clusters_df = pd.read_csv(arg_clusters, sep = '\t')
    plasmid_clusters_df = pd.read_csv(plasmid_clusters, sep = '\t')

    expected_headers_arg_clusters = ['ID', 'Cluster', 'Class', 'Gene']
    expected_headers_plasmid_clusters = ['ID', 'Cluster', 'Replicon', 
                                        'Replicon_group', 'Replicon_subvariant']

    # Check for incorrect headers
    if arg_clusters_df.columns.tolist() != expected_headers_arg_clusters:
        raise HeaderFormatError(f"""
        The values in the header of file '{arg_clusters}' do not match the
        expected values.
        Expected: {expected_headers_arg_clusters}
        Provided: {arg_clusters_df.columns.tolist()}        
        """)

    if plasmid_clusters_df.columns.tolist() != expected_headers_plasmid_clusters:
        raise HeaderFormatError(f"""
        The values in the header of file '{plasmid_clusters}' do not match the
        expected values.
        Expected: {expected_headers_plasmid_clusters}
        Provided: {plasmid_clusters_df.columns.tolist()} 
        """)
    
    return arg_clusters_df, plasmid_clusters_df

def read_counts_file(counts):
    """
    Read the tab-separated table of alignment results produced by running 
    `readcounts.py` and check for the expected formatting.

    :param counts: Path to tab-separated table of alignment results.
    :type counts: str
    :returns counts_df: Counts of mapped reads.
    :rtype counts_df: <class 'pandas.core.frame.DataFrame'>
    :returns num_initial_targets: Number of initial targets included in
    counts_df.
    :rtype num_initial_targets: int
    """

    counts_df = pd.read_csv(counts, sep = '\t')
    expected_headers_counts = ['Gene', 'Hits', 'Gene Fraction']

    # Check for incorrect headers
    if counts_df.columns.tolist() != expected_headers_counts:
        raise HeaderFormatError(f"""
        The values in the header of file '{counts}' do not match the
        expected values.
        Expected: {expected_headers_counts}
        Provided: {counts_df.columns.tolist()} 
        """)

    # Rename the 'Gene' column to 'ID' for downstream merging with cluster
    # dataframes
    counts_df.rename(columns = {'Gene': 'ID'}, inplace = True)

    num_initial_targets = len(counts_df.index)

    return counts_df, num_initial_targets

def parse_flagstat_file(flagstat):
    """
    Parse the file produced by running `samtools flagstat` to obtain the total
    number of reads used for alignment to the bait-capture target database.
    
    :param flagstat: Path to flagstat file.
    :type flagstat: str
    :returns nreads: Total number of reads used for alignment.
    :rtype nreads: int
    """

    with open(flagstat, 'r') as handle:
        nreads = int(handle.readline().split()[0])

    return nreads

def add_prop_reads(counts_df, nreads):
    """
    Add a column to counts_df to represent the proportion of total reads that
    were used for alignment to the number of actual algined reads to each
    target.

    :param counts_df: Counts of mapped reads.
    :type counts_df: <class 'pandas.core.frame.DataFrame'>
    :param nreads: Total number of reads used for alignment.
    :type nreads: int
    :returns counts_df: Counts of mapped reads with proportion of mapped reads.
    :rtype counts_df: <class 'pandas.core.frame.DataFrame'>
    """

    counts_df['Prop_reads'] = counts_df['Hits'].apply(lambda x: float(x) / nreads)

    return counts_df

def filter_counts_df(counts_df, min_prop):
    """
    Filter counts_df based upon the minimum fold-coverage of reads provided by 
    `--min_prop`.

    :param counts_df: Counts of mapped reads.
    :type counts_df: <class 'pandas.core.frame.DataFrame'>
    :param min_prop: Minimum fold-coverage of reads against target required
    for consideration during clustering.
    :type min_prop: float
    :returns counts_df: Filtered counts of mapped reads.
    :rtype counts_df: <class 'pandas.core.frame.DataFrame'>
    :returns num_final_targets: Final number of targets included in counts_df
    after filtering.
    :rtype: int
    """

    counts_df = counts_df[counts_df['Gene Fraction'] >= min_prop]
    num_final_targets = len(counts_df.index)

    return counts_df, num_final_targets

def cluster_merge(arg_clusters_df, plasmid_clusters_df, counts_df):
    """
    Retain only the most relatively abundant target per cluster, and merge the
    ARG and replicon group target read counts into a single dataframe.

    :param arg_clusters_df: ARG clusters.
    :type arg_clusters_df: <class 'pandas.core.frame.DataFrame'>
    :param plasmid_clusters_df: Replicon group clusters.
    :type plasmid_clusters_df: <class 'pandas.core.frame.DataFrame'>
    :param counts_df: Filtered counts of mapped reads.
    :type counts_df: <class 'pandas.core.frame.DataFrame'>
    :returns merged_df: Merged read counts for the most relatively abundant
    ARG and replicon group clusters.
    :rtype merged_df: <class 'pandas.core.frame.DataFrame'>
    """

    counts_arg_clustered_df = pd.merge(arg_clusters_df, counts_df, on = 'ID', how = 'inner')
    counts_arg_clustered_df.drop(arg_clusters_df.columns.drop('Cluster').tolist(), axis = 1, inplace = True)
    counts_arg_clustered_df = counts_arg_clustered_df.groupby('Cluster').aggregate(np.max)
    # print(len(counts_arg_clustered_df.index))

    counts_plasmid_clustered_df = pd.merge(plasmid_clusters_df, counts_df, on = 'ID', how = 'inner')
    counts_plasmid_clustered_df.drop(plasmid_clusters_df.columns.drop('Cluster').tolist(), axis = 1, inplace = True)
    counts_plasmid_clustered_df = counts_plasmid_clustered_df.groupby('Cluster').aggregate(np.max)
    # print(len(counts_plasmid_clustered_df.index))

    merged_df = pd.concat([counts_arg_clustered_df, counts_plasmid_clustered_df])
    # print(len(merged_df.index))

    return merged_df

def save_merged(merged_df, output):

    if not os.path.isfile(output):
        merged_df.to_csv(output, sep = '\t')
    else:
        raise OSError(f"File '{output}' already exists. Unable to save output.")

#-------------------------------------------------------------------------------
# main()
#-------------------------------------------------------------------------------

def main(args):

    counts_df, num_initial_targets = read_counts_file(counts = args.input)

    arg_clusters_df, plasmid_clusters_df = read_cluster_files(
        arg_clusters = args.arg_clusters,
        plasmid_clusters = args.plasmid_clusters)

    # If `-f flagstat` is provided:
    if args.flagstat is not None:
        nreads = parse_flagstat_file(flagstat = args.flagstat)
        counts_df = add_prop_reads(counts_df, nreads)

    counts_df, num_final_targets = filter_counts_df(counts_df, min_prop = args.min_prop)
 
    print(f"""
    Minimum fold-coverage filtering threshold: {args.min_prop}
    Number of targets: {num_initial_targets}
    Number of targets after filtering: {num_final_targets}
    Percentage of targets filtered: {
        np.round(
            (num_initial_targets - num_final_targets) / num_initial_targets * 100, 2
            )
        }%
    """)

    merged_df = cluster_merge(arg_clusters_df, plasmid_clusters_df, counts_df)

    save_merged(merged_df, output = args.output)

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_arguments()
    main(args)