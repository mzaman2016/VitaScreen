"""
IGTD (Image Generator for Tabular Data) - Functions for converting tabular data to images.
Based on Zhu et al. (2021) - Converting tabular data into images for deep learning with CNNs.
"""
from scipy.stats import spearmanr, rankdata
from scipy.spatial.distance import pdist, squareform

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/headless environments
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import time
import pickle as cp


def min_max_transform(data):
    """Linear transformation: min=0, max=1 for each feature."""
    norm_data = np.empty(data.shape)
    norm_data.fill(np.nan)
    for i in range(data.shape[1]):
        v = data[:, i].copy()
        if np.max(v) == np.min(v):
            norm_data[:, i] = 0
        else:
            v = (v - np.min(v)) / (np.max(v) - np.min(v))
            norm_data[:, i] = v
    return norm_data


def generate_feature_distance_ranking(data, method='Pearson'):
    """Generate ranking of distances between features."""
    num = data.shape[1]
    if method == 'Pearson':
        corr = np.corrcoef(np.transpose(data))
    elif method == 'Spearman':
        corr = spearmanr(data).correlation
    elif method == 'Euclidean':
        corr = squareform(pdist(np.transpose(data), metric='euclidean'))
        corr = np.max(corr) - corr
        corr = corr / np.max(corr)
    elif method == 'set':
        corr1 = np.dot(np.transpose(data), data)
        corr2 = data.shape[0] - np.dot(np.transpose(1 - data), 1 - data)
        corr = corr1 / corr2

    corr = 1 - corr
    corr = np.around(a=corr, decimals=10)
    tril_id = np.tril_indices(num, k=-1)
    rank = rankdata(corr[tril_id])
    ranking = np.zeros((num, num))
    ranking[tril_id] = rank
    ranking = ranking + np.transpose(ranking)
    return ranking, corr


def generate_matrix_distance_ranking(num_r, num_c, method='Euclidean'):
    """Calculate ranking of distances between pixels in image grid."""
    for r in range(num_r):
        if r == 0:
            coordinate = np.transpose(np.vstack((np.zeros(num_c), range(num_c))))
        else:
            coordinate = np.vstack((coordinate, np.transpose(np.vstack((np.ones(num_c) * r, range(num_c))))))

    num = num_r * num_c
    cord_dist = np.zeros((num, num))
    if method == 'Euclidean':
        for i in range(num):
            cord_dist[i, :] = np.sqrt(np.square(coordinate[i, 0] * np.ones(num) - coordinate[:, 0]) +
                                     np.square(coordinate[i, 1] * np.ones(num) - coordinate[:, 1]))
    elif method == 'Manhattan':
        for i in range(num):
            cord_dist[i, :] = np.abs(coordinate[i, 0] * np.ones(num) - coordinate[:, 0]) + \
                             np.abs(coordinate[i, 1] * np.ones(num) - coordinate[:, 1])

    tril_id = np.tril_indices(num, k=-1)
    rank = rankdata(cord_dist[tril_id])
    ranking = np.zeros((num, num))
    ranking[tril_id] = rank
    ranking = ranking + np.transpose(ranking)
    coordinate = np.int64(coordinate)
    return (coordinate[:, 0], coordinate[:, 1]), ranking


def IGTD_absolute_error(source, target, max_step=1000, switch_t=0, val_step=50, min_gain=0.00001, random_state=1,
                        save_folder=None, file_name=''):
    """IGTD optimization with absolute error."""
    np.random.seed(random_state)
    if save_folder and os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)

    source = source.copy()
    num = source.shape[0]
    tril_id = np.tril_indices(num, k=-1)
    index = np.array(range(num))
    index_record = np.empty((max_step + 1, num))
    index_record.fill(np.nan)
    index_record[0, :] = index.copy()

    err_v = np.empty(num)
    err_v.fill(np.nan)
    for i in range(num):
        err_v[i] = np.sum(np.abs(source[i, 0:i] - target[i, 0:i])) + \
                   np.sum(np.abs(source[(i + 1):, i] - target[(i + 1):, i]))

    step_record = -np.ones(num)
    err_record = [np.sum(abs(source[tril_id] - target[tril_id]))]
    pre_err = err_record[0]
    t1 = time.time()
    run_time = [0]

    for s in range(max_step):
        delta = np.ones(num) * np.inf
        idr = np.where(step_record == np.min(step_record))[0]
        ii = idr[np.random.permutation(len(idr))[0]]

        for jj in range(num):
            if jj == ii:
                continue
            if ii < jj:
                i, j = ii, jj
            else:
                i, j = jj, ii

            err_ori = err_v[i] + err_v[j] - np.abs(source[j, i] - target[j, i])
            err_i = np.sum(np.abs(source[j, :i] - target[i, :i])) + \
                    np.sum(np.abs(source[(i + 1):j, j] - target[(i + 1):j, i])) + \
                    np.sum(np.abs(source[(j + 1):, j] - target[(j + 1):, i])) + np.abs(source[i, j] - target[j, i])
            err_j = np.sum(np.abs(source[i, :i] - target[j, :i])) + \
                    np.sum(np.abs(source[i, (i + 1):j] - target[j, (i + 1):j])) + \
                    np.sum(np.abs(source[(j + 1):, i] - target[(j + 1):, j])) + np.abs(source[i, j] - target[j, i])
            err_test = err_i + err_j - np.abs(source[i, j] - target[j, i])
            delta[jj] = err_test - err_ori

        delta_norm = delta / (pre_err + 1e-10)
        id_vals = np.where(delta_norm <= switch_t)[0]
        if len(id_vals) > 0:
            jj = np.argmin(delta)
            if ii < jj:
                i, j = ii, jj
            else:
                i, j = jj, ii
            for k in range(num):
                if k < i:
                    err_v[k] = err_v[k] - np.abs(source[i, k] - target[i, k]) - np.abs(source[j, k] - target[j, k]) + \
                               np.abs(source[j, k] - target[i, k]) + np.abs(source[i, k] - target[j, k])
                elif k == i:
                    err_v[k] = np.sum(np.abs(source[j, :i] - target[i, :i])) + \
                        np.sum(np.abs(source[(i + 1):j, j] - target[(i + 1):j, i])) + \
                        np.sum(np.abs(source[(j + 1):, j] - target[(j + 1):, i])) + np.abs(source[i, j] - target[j, i])
                elif k < j:
                    err_v[k] = err_v[k] - np.abs(source[k, i] - target[k, i]) - np.abs(source[j, k] - target[j, k]) + \
                               np.abs(source[k, j] - target[k, i]) + np.abs(source[i, k] - target[j, k])
                elif k == j:
                    err_v[k] = np.sum(np.abs(source[i, :i] - target[j, :i])) + \
                        np.sum(np.abs(source[i, (i + 1):j] - target[j, (i + 1):j])) + \
                        np.sum(np.abs(source[(j + 1):, i] - target[(j + 1):, j])) + np.abs(source[i, j] - target[j, i])
                else:
                    err_v[k] = err_v[k] - np.abs(source[k, i] - target[k, i]) - np.abs(source[k, j] - target[k, j]) + \
                               np.abs(source[k, j] - target[k, i]) + np.abs(source[k, i] - target[k, j])

            ii_v = source[ii, :].copy()
            jj_v = source[jj, :].copy()
            source[ii, :] = jj_v
            source[jj, :] = ii_v
            ii_v = source[:, ii].copy()
            jj_v = source[:, jj].copy()
            source[:, ii] = jj_v
            source[:, jj] = ii_v
            err = delta[jj] + pre_err
            t = index[ii]
            index[ii] = index[jj]
            index[jj] = t
            step_record[ii] = s
            step_record[jj] = s
        else:
            err = pre_err
            step_record[ii] = s

        err_record.append(err)
        if s % 100 == 0:
            print(f'Step {s} err: {err:.2f}')
        index_record[s + 1, :] = index.copy()
        run_time.append(time.time() - t1)

        if s > val_step:
            if np.sum((err_record[-val_step - 1] - np.array(err_record[(-val_step):])) / (err_record[-val_step - 1] + 1e-10) >= min_gain) == 0:
                break
        pre_err = err

    index_record = index_record[:len(err_record), :].astype(np.int64)
    if save_folder:
        pd.DataFrame(index_record).to_csv(os.path.join(save_folder, file_name + '_index.txt'), header=False, index=False, sep='\t')
        pd.DataFrame(np.transpose(np.vstack((err_record, np.array(range(len(err_record)))))), columns=['error', 'steps']).to_csv(
            os.path.join(save_folder, file_name + '_error_and_step.txt'), header=True, index=False, sep='\t')
    return index_record, err_record, run_time


def IGTD(source, target, err_measure='abs', max_step=1000, switch_t=0, val_step=50, min_gain=0.00001, random_state=1,
         save_folder=None, file_name=''):
    """Wrapper for IGTD optimization."""
    return IGTD_absolute_error(source=source, target=target, max_step=max_step, switch_t=switch_t, val_step=val_step,
                               min_gain=min_gain, random_state=random_state, save_folder=save_folder, file_name=file_name)


def generate_image_data(data, index, num_row, num_column, coord, image_folder=None, file_name='', labels=None):
    """
    Generate image data from tabular data using IGTD indices.
    If labels is provided and image_folder set, save as diabetic_X.png and non_diabetic_X.png.
    """
    if isinstance(data, pd.DataFrame):
        samples = data.index.map(str)
        data = data.values
    else:
        samples = [str(i) for i in range(data.shape[0])]

    if image_folder and os.path.exists(image_folder):
        shutil.rmtree(image_folder)
    if image_folder:
        os.makedirs(image_folder, exist_ok=True)

    data_2 = data.copy()
    data_2 = data_2[:, index]
    max_v = np.max(data_2)
    min_v = np.min(data_2)
    data_2 = 255 - (data_2 - min_v) / (max_v - min_v + 1e-10) * 255

    diabetic_count = 0
    non_diabetic_count = 0

    image_data = np.empty((num_row, num_column, data_2.shape[0]))
    image_data.fill(np.nan)
    for i in range(data_2.shape[0]):
        data_i = np.empty((num_row, num_column))
        data_i.fill(np.nan)
        data_i[coord] = data_2[i, :]
        image_data[:, :, i] = data_i
        if image_folder:
            fig = plt.figure()
            plt.imshow(data_i, cmap='gray', vmin=0, vmax=255)
            plt.axis('scaled')
            if labels is not None:
                if labels[i] == 1:
                    img_name = f'diabetic_{diabetic_count}.png'
                    diabetic_count += 1
                else:
                    img_name = f'non_diabetic_{non_diabetic_count}.png'
                    non_diabetic_count += 1
                plt.savefig(fname=os.path.join(image_folder, img_name), bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig(fname=os.path.join(image_folder, f'{file_name}_{samples[i]}_image.png'), bbox_inches='tight', pad_inches=0)
            plt.close(fig)

    return image_data, samples


def save_images_diabetic_naming(image_array, labels, output_dir):
    """
    Save image array to folder with diabetic_X.png and non_diabetic_X.png naming.
    image_array: (H, W, N) - N images
    labels: length N array of 0/1
    Returns: list of (image_filename, label) for each sample
    """
    if os.path.exists(output_dir):
        # Windows-friendly directory removal with retry
        for attempt in range(3):
            try:
                shutil.rmtree(output_dir)
                break
            except (OSError, PermissionError) as e:
                if attempt < 2:
                    time.sleep(0.5)
                else:
                    # If still failing, just try to clear files individually
                    for root, dirs, files in os.walk(output_dir, topdown=False):
                        for name in files:
                            try:
                                os.remove(os.path.join(root, name))
                            except:
                                pass
    os.makedirs(output_dir, exist_ok=True)
    diabetic_count = 0
    non_diabetic_count = 0
    image_labels = []
    for i in range(image_array.shape[2]):
        data_i = np.nan_to_num(image_array[:, :, i], nan=0.0)
        data_i = np.clip(data_i, 0, 255)
        fig = plt.figure()
        plt.imshow(data_i, cmap='gray', vmin=0, vmax=255)
        plt.axis('scaled')
        if labels[i] == 1:
            img_name = f'diabetic_{diabetic_count}.png'
            diabetic_count += 1
        else:
            img_name = f'non_diabetic_{non_diabetic_count}.png'
            non_diabetic_count += 1
        plt.savefig(fname=os.path.join(output_dir, img_name), bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        image_labels.append({'image_file': img_name, 'label': int(labels[i])})
    return image_labels


def table_to_image(norm_d, scale, fea_dist_method, image_dist_method, save_image_size, max_step, val_step, normDir,
                   error, switch_t=0, min_gain=0.00001):
    """Convert tabular data to images using IGTD algorithm."""
    os.makedirs(normDir, exist_ok=True)

    ranking_feature, _ = generate_feature_distance_ranking(data=norm_d, method=fea_dist_method)
    fig = plt.figure(figsize=(save_image_size, save_image_size))
    plt.imshow(np.max(ranking_feature) - ranking_feature, cmap='gray', interpolation='nearest')
    plt.savefig(fname=os.path.join(normDir, 'original_feature_ranking.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    coordinate, ranking_image = generate_matrix_distance_ranking(num_r=scale[0], num_c=scale[1], method=image_dist_method)
    fig = plt.figure(figsize=(save_image_size, save_image_size))
    plt.imshow(np.max(ranking_image) - ranking_image, cmap='gray', interpolation='nearest')
    plt.savefig(fname=os.path.join(normDir, 'image_ranking.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    save_sub = os.path.join(normDir, error)
    index, err, run_time = IGTD(source=ranking_feature, target=ranking_image, err_measure='abs',
        max_step=max_step, switch_t=switch_t, val_step=val_step, min_gain=min_gain, random_state=1,
        save_folder=save_sub, file_name='')

    fig = plt.figure()
    plt.plot(run_time, err)
    plt.savefig(fname=os.path.join(normDir, 'error_and_runtime.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    fig = plt.figure()
    plt.plot(range(len(err)), err)
    plt.savefig(fname=os.path.join(normDir, 'error_and_iteration.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    min_id = np.argmin(err)
    data, samples = generate_image_data(data=norm_d, index=index[min_id, :], num_row=scale[0], num_column=scale[1],
        coord=coordinate, image_folder=os.path.join(normDir, 'data'), file_name='')

    with open(os.path.join(normDir, 'Results.pkl'), 'wb') as f:
        cp.dump(norm_d, f)
        cp.dump(data, f)
        cp.dump(samples, f)

    with open(os.path.join(normDir, 'Results_Auxiliary.pkl'), 'wb') as f:
        cp.dump(ranking_feature, f)
        cp.dump(ranking_image, f)
        cp.dump(coordinate, f)
        cp.dump(err, f)
        cp.dump(run_time, f)
        cp.dump(index[min_id, :], f)  # Save optimal index for reuse on test data

    return data, samples, index[min_id, :]


def generate_images_with_index(data_df, feature_cols, index, num_row, num_col, output_dir=None, labels=None):
    """
    Generate IGTD images using pre-computed index (e.g., from training data).
    Use this for test data to avoid running IGTD optimization again.
    If labels provided, saves as diabetic_X.png and non_diabetic_X.png.
    """
    X = data_df[feature_cols].values
    X_norm = min_max_transform(X)
    norm_data = pd.DataFrame(X_norm, columns=feature_cols, index=data_df.index)

    coordinate, _ = generate_matrix_distance_ranking(num_r=num_row, num_c=num_col, method='Euclidean')
    image_data, samples = generate_image_data(
        data=norm_data, index=index, num_row=num_row, num_column=num_col,
        coord=coordinate, image_folder=output_dir, file_name='', labels=labels
    )
    return image_data, samples
