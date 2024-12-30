import os

def compile(dice, hd, center, test_center, logger, dst='results_folder'):
    # write documentation
    """
    Compile the results of the training process.

    Parameters
    ----------
    dice : tuple
        Tuple with the (dice, dice_std).
    hd : tuple
        Tuple with the (hd, hd_std).
    center : int
        Id of the center used for training.
    test_center : int
        Id of the center used for testing.
    dst : str
        Destination folder to save the results.

    Returns
    -------
    None
    """

    if test_center == 'all' or test_center == -1:
        test_center = 'all'
        file_name = 'compiled_results_all.csv'
    else:
        file_name = 'compiled_results_detailed.csv'

    file_path = os.path.join(dst, file_name)

    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write('train_center,test_center,dice,dice_std,hd,hd_std\n')

    with open(file_path, 'a') as f:
        f.write(f'{center},{test_center},{dice[0]:.2f},{dice[1]:.2f},{hd[0]:.2f},{hd[1]:.2f}\n')

    logger.print_to_log_file(f'Compiled results saved to {file_path}')
    return None