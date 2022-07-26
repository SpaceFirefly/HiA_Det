"""
  @Author  : Xinyang Li
  @Time    : 2022/5/6 上午11:04
"""
import os.path
import random
import shutil
import subprocess

def recreate_dir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)

def skymaker(config, outdir, N):
    for i in range(N):
        img_name = os.path.join(outdir, '{}.fits'.format(i))
        exposure = '{}.0'.format(int(random.random() * 600 + 300))         # 300~900
        mag_limit_l = int(random.random() * 3 + 10)         # 10~13
        mag_limit_r = mag_limit_l + int(random.random() * 4 + 6)
        mag_limit_range = '{}.0,{}.0'.format(mag_limit_l, mag_limit_r)
        mag_zero = mag_limit_r
        mag_back_bias = int(random.random() * 3)
        mag_back = mag_limit_r - mag_back_bias
        sky_cmd = 'sky -c {0} -IMAGE_NAME {1} -EXPOSURE_TIME {2} -MAG_ZEROPOINT {3} -BACK_MAG {4} -MAG_LIMITS {5}'.format(
            config, img_name, exposure, mag_zero, mag_back, mag_limit_range)

        process = subprocess.Popen(sky_cmd, shell=True)
        process.wait()

def moveSkyResults(dir, im_size=256):
    recreate_dir(os.path.join(dir, 'images'))
    recreate_dir(os.path.join(dir, 'lists'))
    for fname in os.listdir(dir):
        if fname.split('.')[-1] == 'list':
            f_in = open(os.path.join(dir, fname), 'r')
            f_out = open(os.path.join(dir, 'lists', fname), 'w')
            while True:
                line = f_in.readline()
                if line:
                    line_split = line.split()
                    cen_x, cen_y = line_split[1:3]
                    cen_x = float(cen_x)
                    cen_y = float(cen_y)
                    if 0 < cen_x < im_size and 0 < cen_y < im_size:
                        f_out.write(line)
                else:
                    break
            f_in.close()
            f_out.close()
            os.remove(os.path.join(dir, fname))
        elif fname.split('.')[-1] == 'fits':
            shutil.move(os.path.join(dir, fname), os.path.join(dir, 'images'))
    assert len([f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]) == 0


if __name__ == '__main__':
    config = 'default.conf'
    outdir = os.path.join('/home/lixy/workspace/dataset/starSimu_input')
    recreate_dir(outdir)

    ''' ===== 1. build starSimu_input ===== '''
    # # train
    # outdir_train = os.path.join(outdir, 'train')
    # recreate_dir(outdir_train)
    # skymaker(config, outdir_train, 256)
    # moveSkyResults(outdir_train)
    # print('train dataset build done!')
    # val
    outdir_val = os.path.join(outdir, 'val')
    recreate_dir(outdir_val)
    skymaker(config, outdir_val, 10)
    moveSkyResults(outdir_val)
    print('val dataset build done!')

    ''' ===== 2. build starSimu ==== '''


