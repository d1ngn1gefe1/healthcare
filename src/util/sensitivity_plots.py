'''
Generates threshold vs detection rate sensitivity plots
'''
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

DATA_FILE = 'rtw_itop_front.tsv'

def accepts(**types):
    """
    Python function decorator to check number and type of input parameters.
    Throws an exception if there is a type mismatch between a function's
    input parameters and the declared parameter types.
    """
    def check_accepts(f):
        assert len(types) == f.func_code.co_argcount, \
        '''Number of input parameters does not match "%s"''' % f.func_name
        def new_f(*args, **kwds):
            for i, v in enumerate(args):
                # Don't check the type of 'self' from class functions
                if f.func_code.co_varnames[i] == 'self':
                    continue
                if types.has_key(f.func_code.co_varnames[i]) and \
                    not isinstance(v, types[f.func_code.co_varnames[i]]):
                    raise Exception('''Type mismatch: '%s'=%r should be %s''' %\
                                    (f.func_code.co_varnames[i], v, \
                                    types[f.func_code.co_varnames[i]]))

            for k, v in kwds.iteritems():
                if types.has_key(k) and not isinstance(v, types[k]):
                    raise Exception("Arg '%s'=%r does not match %s" % \
                        (k, v, types[k]))

            return f(*args, **kwds)
        new_f.func_name = f.func_name
        return new_f
    return check_accepts

def main(**kwargs):
    '''
    Main entry point of the program
    '''
    # Get the GT joint positions to compute PCKh

    # Get the actual localization error to compute detection rates
    data = pd.read_csv(DATA_FILE, sep='\t')
    # ITOP
    visible_joints = ['H','N','LS','RS','LE','RE','LH','RH','T','LHIP','RHIP','LK','RK','LF','RF']
    # OTOP
    #visible_joints = ['H', 'C', 'LS', 'LE', 'LH', 'RS', 'RE', 'RH']
    # EVAL
    # visible_joints = ['N','H','LS','LE','LH','RS','RE','RH','LK','LF','RK','RF']
    thresholds = np.linspace(0, 20, num=201)
    detection_rates = np.zeros((thresholds.shape[0], len(visible_joints)))
    for j, jname in enumerate(visible_joints):
        errors = np.array(data[jname])
        detection_rates[:,j] = compute_detection_curve(errors, thresholds)
        plt.plot(thresholds, detection_rates[:,j], linewidth=2)

    plt.tick_params(axis='x', colors='black')
    plt.tick_params(axis='y', colors='black')
    plt.ylabel('Detection Rate', color='black')
    plt.xlabel('Precision Threshold (cm)', color='black')
    plt.ylim([0, 1])
    plt.legend(visible_joints, loc='lower right', prop={'size': 14})
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    plt.savefig('itop_rtw.eps', format='eps')
    plt.show()

    print 'Threshold of 10 cm'
    print detection_rates[np.where(thresholds == 10)[0]]

@accepts(errors=np.ndarray, thresholds=np.ndarray)
def compute_detection_curve(errors, thresholds):
    '''
    Returns a (num_threholds, 1) array of detection rates for each thresholds
    '''
    num_thresholds = thresholds.shape[0]
    num_frames = errors.shape[0]
    detection_rates = np.zeros((num_thresholds,))
    for i in xrange(num_thresholds):
        num_correct = np.sum(errors < thresholds[i])
        detection_rates[i] = 1.0 * num_correct / num_frames
    return detection_rates

if __name__ == '__main__':
    # Configure matplotlib
    plt.style.use('ggplot')
    matplotlib.rcParams.update({'font.size': 22})
    # Parse command line arguments
    P = argparse.ArgumentParser()
    cmd_args = P.parse_args()
    main(**vars(cmd_args))
