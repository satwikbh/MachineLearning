from avclass_common import AvLabels
from operator import itemgetter
from pymongo import MongoClient
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil

import argparse
import traceback
import os
import urllib
import evaluate_clustering as ec
import sys

sys.path.insert(0, 'lib/')

config = ConfigUtil.get_config_instance()
log = LoggerUtil("").get()

# Default alias file
default_alias_file = "data/default.aliases"
# Default generic tokens file
default_gen_file = "data/default.generics"


def get_client(address, port, username, password, auth_db, is_auth_enabled):
    try:
        if is_auth_enabled:
            client = MongoClient("mongodb://" + username + ":" + password + "@" + address + ":" + port + "/" + auth_db)
        else:
            client = MongoClient("mongodb://" + address + ":" + port + "/" + auth_db)
        return client
    except Exception as e:
        print("Error", e)


def get_connection():
    username = config['environment']['mongo']['username']
    pwd = config['environment']['mongo']['password']
    password = urllib.quote(pwd)
    address = config['environment']['mongo']['address']
    port = config['environment']['mongo']['port']
    auth_db = config['environment']['mongo']['auth_db']
    is_auth_enabled = config['environment']['mongo']['is_auth_enabled']

    client = get_client(address=address, port=port, auth_db=auth_db, is_auth_enabled=is_auth_enabled,
                        username=username, password=password)

    db_name = config['environment']['mongo']['db_name']
    vt_collection_name = config['environment']['mongo']['virus_total_collection_name']
    avclass_collection_name = config['environment']['mongo']['avclass_collection_name']

    db = client[db_name]
    vt_collection = db[vt_collection_name]
    avclass_collection = db[avclass_collection_name]
    return client, vt_collection, avclass_collection


def guess_hash(h):
    """
    Given a hash string, guess the hash type based on the string length
    :param h:
    :return:
    """
    hlen = len(h)
    if hlen == 32:
        return 'md5'
    elif hlen == 40:
        return 'sha1'
    elif hlen == 64:
        return 'sha256'
    else:
        return None


def main(args):
    # Select hash used to identify sample, by default MD5
    hash_type = args.hash if args.hash else 'md5'

    # If ground truth provided, read it from file
    gt_dict = {}
    if args.gt:
        with open(args.gt, 'r') as gt_fd:
            for line in gt_fd:
                gt_hash, family = map(str.lower, line.strip().split('\t', 1))
                gt_dict[gt_hash] = family

        # Guess type of hash in ground truth file
        hash_type = guess_hash(gt_dict.keys()[0])

    # Create AvLabels object
    av_labels = AvLabels(args.gen, args.alias, args.av)

    # Select input file with AV labels
    ifile = args.vt if args.vt else args.lb

    # If verbose, open log file
    if args.verbose:
        log_filename = os.path.basename(os.path.splitext(ifile)[0]) + \
                       '.verbose'
        verb_fd = open(log_filename, 'w+')

    # Process each JSON
    vt_all = 0
    vt_empty = 0
    singletons = 0

    client, vt_collection, avclass_collection = get_connection()
    cursor = avclass_collection.aggregate([{"$group": {"_id": "$md5"}}])

    list_of_md5 = list()
    for each in cursor:
        list_of_md5.append(each["_id"])

    log.info("Total number of keys : {}".format(len(list_of_md5)))

    first_token_dict = {}
    token_count_map = {}
    pair_count_map = {}
    token_family_map = {}
    fam_stats = {}

    counter = len(list_of_md5)
    count, index = 0, 0
    batch_size = 1000
    while count + batch_size <= counter:
        print("Iteration : #{}".format(index))
        if count + batch_size <= counter:
            keys = list_of_md5[count: count + batch_size]
        else:
            keys = list_of_md5[count:]
        p_cursor = avclass_collection.find({"md5": {"$in": keys}})
        for vt_rep in p_cursor:
            try:
                vt_rep.pop("_id")
                vt_all += 1
                sample_info = av_labels.get_sample_info(vt_rep, args.vt)
                if sample_info is None:
                    try:
                        name = vt_rep['md5']
                        sys.stderr.write('\nNo AV labels for %s\n' % name)
                    except KeyError:
                        sys.stderr.write('\nCould not process: %s\n' % vt_all)
                    sys.stderr.flush()
                    vt_empty += 1
                    continue

                # Sample's name is selected hash type (md5 by default)
                name = getattr(sample_info, hash_type)

                # If the VT report has no AV labels, continue
                if not sample_info[3]:
                    vt_empty += 1
                    sys.stderr.write('\nNo AV labels for %s\n' % name)
                    sys.stderr.flush()
                    continue

                # Get the distinct tokens from all the av labels in the report
                # And print them. If not verbose, print the first token.
                # If verbose, print the whole list
                try:
                    # Get distinct tokens from AV labels
                    tokens = av_labels.get_family_ranking(sample_info).items()

                    # If alias detection, populate maps
                    if args.aliasdetect:
                        prev_tokens = set([])
                        for entry in tokens:
                            curr_tok = entry[0]
                            curr_count = token_count_map.get(curr_tok)
                            if curr_count:
                                token_count_map[curr_tok] = curr_count + 1
                            else:
                                token_count_map[curr_tok] = 1
                            for prev_tok in prev_tokens:
                                if prev_tok < curr_tok:
                                    pair = (prev_tok, curr_tok)
                                else:
                                    pair = (curr_tok, prev_tok)
                                pair_count = pair_count_map.get(pair)
                                if pair_count:
                                    pair_count_map[pair] = pair_count + 1
                                else:
                                    pair_count_map[pair] = 1
                            prev_tokens.add(curr_tok)

                    # If generic token detection, populate map
                    if args.gendetect and args.gt:
                        for entry in tokens:
                            curr_tok = entry[0]
                            curr_fam_set = token_family_map.get(curr_tok)
                            family = gt_dict[name] if name in gt_dict else None
                            if curr_fam_set and family:
                                curr_fam_set.add(family)
                            elif family:
                                token_family_map[curr_tok] = set(family)

                    # Top candidate is most likely family name
                    if tokens:
                        family = tokens[0][0]
                        is_singleton = False
                    else:
                        family = "SINGLETON:" + name
                        is_singleton = True
                        singletons += 1

                    # Check if sample is PUP, if requested
                    if args.pup:
                        is_pup = av_labels.is_pup(sample_info[3])
                        if is_pup:
                            is_pup_str = "\t1"
                        else:
                            is_pup_str = "\t0"
                    else:
                        is_pup = None
                        is_pup_str = ""

                    # Build family map for precision, recall, computation
                    first_token_dict[name] = family

                    # Get ground truth family, if available
                    if args.gt:
                        gt_family = '\t' + gt_dict[name] if name in gt_dict else ""
                    else:
                        gt_family = ""

                    # Print family (and ground truth if available) to stdout
                    # print '%s\t%s%s%s' % (name, family, gt_family, is_pup_str)

                    avclass_results = dict()
                    avclass_results["result"] = family
                    avclass_results["verbose"] = tokens

                    avclass_collection.update_one(
                        {'md5': name},
                        {"$set": {'avclass': avclass_results}}
                    )

                    # If verbose, print tokens (and ground truth if available)
                    # to log file
                    if args.verbose:
                        verb_fd.write('%s\t%s%s%s\n' % (name, tokens, gt_family, is_pup_str))

                    # Store family stats (if required)
                    if args.fam:
                        if is_singleton:
                            ff = 'SINGLETONS'
                        else:
                            ff = family
                        try:
                            numAll, numMal, numPup = fam_stats[ff]
                        except KeyError:
                            numAll = 0
                            numMal = 0
                            numPup = 0

                        numAll += 1
                        if args.pup:
                            if is_pup:
                                numPup += 1
                            else:
                                numMal += 1
                        fam_stats[ff] = (numAll, numMal, numPup)
                except:
                    traceback.print_exc(file=sys.stderr)
                    continue
            except Exception as e:
                log.error("ERROR : {}\nindex : {}\t each_md5 : {}".format(e, index, vt_rep))
        count += batch_size
        index += 1

    # Debug info
    sys.stderr.write('\r[-] %d JSON read' % vt_all)
    sys.stderr.flush()
    sys.stderr.write('\n')

    # Print statistics
    sys.stderr.write(
        "[-] Samples: %d NoLabels: %d Singletons: %d "
        "GroundTruth: %d\n" % (
            vt_all, vt_empty, singletons, len(gt_dict)))

    # If ground truth, print precision, recall, and F1-measure
    if args.gt and args.eval:
        precision, recall, fmeasure = \
            ec.eval_precision_recall_fmeasure(gt_dict,
                                              first_token_dict)
        sys.stderr.write("Precision: %.2f\tRecall: %.2f\tF1-Measure: %.2f\n" %
                         (precision, recall, fmeasure))

    # If generic token detection, print map
    if args.gendetect:
        # Open generic tokens file
        gen_filename = os.path.basename(os.path.splitext(ifile)[0]) + \
                       '.gen'
        gen_fd = open(gen_filename, 'w+')
        # Output header line
        gen_fd.write("Token\t#Families\n")
        sorted_pairs = sorted(token_family_map.iteritems(),
                              key=lambda x: len(x[1]) if x[1] else 0,
                              reverse=True)
        for (t, fset) in sorted_pairs:
            gen_fd.write("%s\t%d\n" % (t, len(fset)))

        # Close generic tokens file
        gen_fd.close()

    # If alias detection, print map
    if args.aliasdetect:
        # Open alias file
        alias_filename = os.path.basename(os.path.splitext(ifile)[0]) + \
                         '.alias'
        alias_fd = open(alias_filename, 'w+')
        # Sort token pairs by number of times they appear together
        sorted_pairs = sorted(
            pair_count_map.items(), key=itemgetter(1))
        # Output header line
        alias_fd.write("# t1\tt2\t|t1|\t|t2|\t|t1^t2|\t|t1^t2|/|t1|\n")
        # Compute token pair statistic and output to alias file
        for (t1, t2), c in sorted_pairs:
            n1 = token_count_map[t1]
            n2 = token_count_map[t2]
            if n1 < n2:
                x = t1
                y = t2
                xn = n1
                yn = n2
            else:
                x = t2
                y = t1
                xn = n2
                yn = n1
            f = float(c) / float(xn)
            alias_fd.write("%s\t%s\t%d\t%d\t%d\t%0.2f\n" % (
                x, y, xn, yn, c, f))
        # Close alias file
        alias_fd.close()

    # If family statistics, output to file
    if args.fam:
        # Open family file
        fam_filename = os.path.basename(os.path.splitext(ifile)[0]) + \
                       '.families'
        fam_fd = open(fam_filename, 'w+')
        # Output header line
        if args.pup:
            fam_fd.write("# Family\tTotal\tMalware\tPUP\tFamType\n")
        else:
            fam_fd.write("# Family\tTotal\n")
        # Sort map
        sorted_pairs = sorted(fam_stats.items(), key=itemgetter(1),
                              reverse=True)
        # Print map contents
        for (f, fstat) in sorted_pairs:
            if args.pup:
                if fstat[1] > fstat[2]:
                    famType = "malware"
                else:
                    famType = "pup"
                fam_fd.write("%s\t%d\t%d\t%d\t%s\n" % (f, fstat[0], fstat[1],
                                                       fstat[2], famType))
            else:
                fam_fd.write("%s\t%d\n" % (f, fstat[0]))
        # Close file
        fam_fd.close()

    # Close log file
    if args.verbose:
        sys.stderr.write('[-] Verbose output in %s\n' % log_filename)
        verb_fd.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(prog='avclass_labeler',
                                        description='''Extracts the family of a set of samples.
            Also calculates precision and recall if ground truth available''')

    argparser.add_argument('-vt',
                           help='file with full VT reports '
                                '(REQUIRED if -lb argument not present)')

    argparser.add_argument('-lb',
                           help='file with simplified JSON reports'
                                '{md5,sha1,sha256,scan_date,av_labels} '
                                '(REQUIRED if -vt not present)')

    argparser.add_argument('-gt',
                           help='file with ground truth')

    argparser.add_argument('-eval',
                           action='store_true',
                           help='if used it evaluates clustering accuracy.'
                                ' Prints precision, recall, F1-measure. Requires -gt parameter')

    argparser.add_argument('-alias',
                           help='file with aliases.',
                           default=default_alias_file)

    argparser.add_argument('-gen',
                           help='file with generic tokens.',
                           default=default_gen_file)

    argparser.add_argument('-av',
                           help='file with list of AVs to use')

    argparser.add_argument('-pup',
                           action='store_true',
                           help='if used each sample is classified as PUP or not')

    argparser.add_argument('-gendetect',
                           action='store_true',
                           help='if used produce generics file at end. Requires -gt parameter')

    argparser.add_argument('-aliasdetect',
                           action='store_true',
                           help='if used produce aliases file at end')

    argparser.add_argument('-v', '--verbose',
                           action='store_true',
                           help='output .verbose file with distinct tokens')

    argparser.add_argument('-hash',
                           help='hash used to name samples. Should match ground truth',
                           choices=['md5', 'sha1', 'sha256'])

    argparser.add_argument('-fam',
                           action='store_true',
                           help='if used produce families file with PUP/malware counts per family')

    args = argparser.parse_args()

    if not args.vt and not args.lb:
        sys.stderr.write('Argument -vt or -lb is required\n')
        exit(1)

    if args.vt and args.lb:
        sys.stderr.write('Use either -vt or -lb argument, not both.\n')
        exit(1)

    if args.gendetect and not args.gt:
        sys.stderr.write('Generic token detection requires -gt param\n')
        exit(1)

    if args.eval and not args.gt:
        sys.stderr.write('Evaluating clustering accuracy needs -gt param\n')
        exit(1)

    if args.alias:
        if args.alias == '/dev/null':
            sys.stderr.write('[-] Using no aliases\n')
        else:
            sys.stderr.write('[-] Using aliases in %s\n' % (
                args.alias))
    else:
        sys.stderr.write('[-] Using generic aliases in %s\n' % (
            default_alias_file))

    if args.gen:
        if args.gen == '/dev/null':
            sys.stderr.write('[-] Using no generic tokens\n')
        else:
            sys.stderr.write('[-] Using generic tokens in %s\n' % (
                args.gen))
    else:
        sys.stderr.write('[-] Using default generic tokens in %s\n' % (
            default_gen_file))

    main(args)