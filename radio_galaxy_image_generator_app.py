import streamlit as st
import os
import numpy as np
from torchvision.datasets.utils import download_url
import torch
import wGAN_models as models
import torchvision.transforms as transforms
import h5py
import random
from firstgalaxydata import FIRSTGalaxyData

st.set_page_config(layout="wide", page_title="# Radio Galaxy Source Generation for classes FRI, FRII, Compact and Bent", page_icon=":taxi:")

if __name__ == "__main__":
    print("app started")
    device = torch.device('cpu')

    @st.cache
    def get_meta_data():
         metadata = {
             'image_shape': [1, 128, 128],
             'nz': 100,
             'ngf': 64,
             'bsize': 10,
             'num_workers': 2}
         return metadata


    metadata = get_meta_data()

    @st.cache
    def download():
        urls = {
            "generator_epoch_3361_iter_30250_cla0.pt": "https://syncandshare.desy.de/index.php/s/XJbDAn2yGGipzDJ/download",
            "generator_epoch_3833_iter_34500_cla1.pt": "https://syncandshare.desy.de/index.php/s/ArwnY6AetKPa9SM/download",
            "generator_epoch_1305_iter_11750_cla2.pt": "https://syncandshare.desy.de/index.php/s/PQX6KrDYGjfnqRR/download",
            "generator_epoch_4416_iter_39750_cla3.pt": "https://syncandshare.desy.de/index.php/s/HDMKN5fCp5okNBo/download",
        }
        # download file
        for key in urls.keys():
            download_url(urls[key], os.getcwd(), key)

    download()

    @st.experimental_memo(suppress_st_warning=True)
    def get_generators(metadata):
        #st.write("Cache miss: get_generators")
        checkpoints = {
            0: "generator_epoch_3361_iter_30250_cla0.pt",
            1: "generator_epoch_3833_iter_34500_cla1.pt",
            2: "generator_epoch_1305_iter_11750_cla2.pt",
            3: "generator_epoch_4416_iter_39750_cla3.pt"
        }
        generators = {k: models.Generator(metadata['nz'], 1, metadata['ngf'], 4).cpu() for k in checkpoints}
        for label, checkpoint in checkpoints.items():
            generators[label].load_state_dict(torch.load(checkpoint, map_location=device))
            generators[label].eval();

        return generators

    generators = get_generators(metadata)

    @st.experimental_memo(suppress_st_warning=True)
    def get_generated_images(_generators, bsize, n_gen_images, nz, device):
        #st.write("Cache miss: get_generated_images")
        """
        n_gen_images has to be divisible by n_generators
        """

        tensor_opt = {'device': device, 'dtype': torch.float, 'requires_grad': False}
        onehot = torch.zeros(4, 4, **tensor_opt)
        onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3]).view(4, 1).to(device), 1).view(4, 4, 1, 1)

        generated_images = {k: torch.Tensor() for k in generators}
        gen_images_per_gen = n_gen_images // len(generators)
        for label_ind in generators.keys():
            img_counter = 0
            while img_counter < gen_images_per_gen:
                noise = torch.randn(bsize, nz, 1, 1, device=device, requires_grad=False)
                labels = torch.tensor([label_ind] * bsize, device=device, requires_grad=False)
                with torch.no_grad():
                    res = ((generators[label_ind](noise, onehot[labels]) / 2 + .5) * 255).int().cpu()
                    expected = img_counter + bsize
                    #                     print(expected)
                    if expected > gen_images_per_gen:
                        res = res[:gen_images_per_gen - img_counter]
                    generated_images[label_ind] = res if generated_images[label_ind].shape[0] == 0 else torch.cat(
                        (generated_images[label_ind], res))
                    img_counter += bsize
        #                 print(generated_images[label_ind].shape[0])
        return generated_images


    @st.experimental_singleton(suppress_st_warning=True)
    def get_data_loader():
        #st.write("Cache miss: get_data_loader")
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5]),
             transforms.Lambda(lambda x: ((x / 2 + .5) * 255).int()),
             transforms.CenterCrop([metadata["image_shape"][1], metadata["image_shape"][2]])
             ])

        real_data = FIRSTGalaxyData(root=os.getcwd(),
                                    selected_classes=["FRI", "FRII", "Compact", "Bent"], transform=train_transform,
                                    selected_split="test",
                                    input_data_list=["galaxy_data_crossvalid_test_h5.h5"])
        print(real_data)
        dataloader = torch.utils.data.DataLoader(real_data, batch_size=metadata['bsize'], shuffle=False,
                                                 num_workers=metadata['num_workers'])
        return dataloader


    dataloader = get_data_loader()
    
    st.write("# Morphological Classification of Radio Galaxies with wGAN-supported Augmentation")
    st.write("Paper: https://arxiv.org/abs/2212.08504")
    seed = st.slider('seed', 1, 10, 1)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    num_gen_img = st.slider('Num of generated images', 1, 100, 50)
    res = get_generated_images(generators, metadata['nz'], num_gen_img, metadata['nz'], device)

    #@st.experimental_memo(suppress_st_warning=True)
    def get_image_stacks():
        #st.write("Cache miss: get_image_stacks")
        images = []
        labels = []

        for imgs, labels_ in dataloader:
            images.extend(imgs)
            labels.extend(labels_)

        labels = torch.vstack(labels).flatten()
        images = torch.vstack(images)
        return images,labels

    images, labels = get_image_stacks()


    @st.experimental_memo(suppress_st_warning=True)
    def get_image_stack_gen(_result):
        #st.write("Cache miss: get_image_stack_gen")
        gen_imgs = {k: torch.vstack([z for z in v]) for k,v in _result.items()}
        return gen_imgs

    gen_imgs = get_image_stack_gen(res)

    sl1,sl2,sl3,sl4 = st.columns((1, 1, 1, 1))
    with sl1:
        st.write("## FRI")
        fri_thres = st.slider('FRI threshold', 1, 15000, 1) #7500
    with sl2:
        st.write("## FRII")
        frii_thres = st.slider('FRII threshold', 1, 15000, 1) #7500
    with sl3:
        st.write("## Compact")
        compact_thres = st.slider('Compact threshold', 1, 5000, 1) #2500
    with sl4:
        st.write("## Bent")
        bent_thres = st.slider('Bent threshold', 1, 15000, 1) #7500

    thresholds = {
        0: fri_thres,  # 15000
        1: frii_thres,  # , 750015000
        2: compact_thres,  # 2500,5000
        3: bent_thres  # 7500,15000
    }
    real_sums = {k: images[labels==k].sum((1,2)) for k in range(4)}
    gen_sums = {k: v.sum((1,2)) for k,v in gen_imgs.items()}

    thresh_real_images = {k: images[labels == k][real_sums[k] > thresholds[k]] for k in range(4)}
    thresh_gen_images = {k: gen_imgs[k][gen_sum > thresholds[k]] for k, gen_sum in gen_sums.items()}

    del real_sums
    del gen_sums
    del gen_imgs

    thresh_real_indices = {k: range(len(thresh_real_images[k])) for k in range(4)}
    thresh_gen_indices = {k: range(len(thresh_gen_images[k])) for k in range(4)}

    # real
    #idx = torch.randperm(thresh_real_images[0].shape[0])
    tr0 = thresh_real_images[0][thresh_real_indices[0]].view(thresh_real_images[0].size())

    #idx = torch.randperm(thresh_real_images[1].shape[0])
    tr1 = thresh_real_images[1][thresh_real_indices[1]].view(thresh_real_images[1].size())

    #idx = torch.randperm(thresh_real_images[2].shape[0])
    tr2 = thresh_real_images[2][thresh_real_indices[2]].view(thresh_real_images[2].size())

    #idx = torch.randperm(thresh_real_images[3].shape[0])
    tr3 = thresh_real_images[3][thresh_real_indices[3]].view(thresh_real_images[3].size())

    # gen
    #idx = torch.randperm(thresh_gen_images[0].shape[0])
    tg0 = thresh_gen_images[0][thresh_gen_indices[0]].view(thresh_gen_images[0].size())

    #idx = torch.randperm(thresh_gen_images[1].shape[0])
    tg1 = thresh_gen_images[1][thresh_gen_indices[1]].view(thresh_gen_images[1].size())

    #idx = torch.randperm(thresh_gen_images[2].shape[0])
    tg2 = thresh_gen_images[2][thresh_gen_indices[2]].view(thresh_gen_images[2].size())

    #idx = torch.randperm(thresh_gen_images[3].shape[0])
    tg3 = thresh_gen_images[3][thresh_gen_indices[3]].view(thresh_gen_images[3].size())


    #@st.experimental_memo(suppress_st_warning=True)
    def convert_data_to_hdf5(d, h):
        filename="{}.h5".format(h)
        hf = h5py.File(filename, "w")
        hf.create_dataset("generated_radio_galaxy_data_FRI", data=d)
        hf.close()
        return filename


    FRI_real_col, FRI_gen_col, FRII_real_col, FRII_gen_col, Compact_real_col, Compact_gen_col ,\
        Bent_real_col, Bent_gen_col= st.columns((1,1, 1,1, 1,1, 1,1))


    with FRI_real_col:
        st.write("### FRI real:")
        fri_real_ind = st.slider("FRI real index", 0, len(thresh_real_images[0])-1, 0)
        fri_real_img = tr0[fri_real_ind].cpu().detach().numpy()
        st.image(fri_real_img, use_column_width=True)
    with FRI_gen_col:
        st.write("### FRI gen:")
        fri_gen_ind = st.slider("FRI gen index", 0, len(thresh_gen_images[0])-1, 0)
        fri_gen_img = tg0[fri_gen_ind].cpu().detach().numpy()
        st.image(fri_gen_img, use_column_width=True)
        filename = convert_data_to_hdf5(tg0, random.getrandbits(128))
        with open(filename, "rb") as f:
            st.download_button("Download generated data", file_name=filename, data=f, mime='application/x-hdf5')
    with FRII_real_col:
        st.write("### FRII real:")
        frii_real_ind = st.slider("FRII real index", 0, len(thresh_real_images[1])-1,0)
        frii_real_img = tr1[frii_real_ind].cpu().detach().numpy()
        st.image(frii_real_img, use_column_width=True)
    with FRII_gen_col:
        st.write("### FRII gen:")
        frii_gen_ind = st.slider("FRII gen index", 0, len(thresh_gen_images[1])-1,0)
        frii_gen_img = tg1[frii_gen_ind].cpu().detach().numpy()
        st.image(frii_gen_img, use_column_width=True)
        filename = convert_data_to_hdf5(tg1, random.getrandbits(128))
        with open(filename, "rb") as f:
            st.download_button("Download generated data", file_name=filename, data=f, mime='application/x-hdf5')
    with Compact_real_col:
        st.write("### Compact real:")
        compact_real_ind = st.slider("Compact real index", 0, len(thresh_real_images[2])-1,0)
        compact_real_img = tr2[compact_real_ind].cpu().detach().numpy()
        st.image(compact_real_img, use_column_width=True)
    with Compact_gen_col:
        st.write("### Compact gen:")
        compact_gen_ind = st.slider("Compact gen index", 0, len(thresh_gen_images[2])-1,0)
        compact_gen_img = tg2[compact_gen_ind].cpu().detach().numpy()
        st.image(compact_gen_img, use_column_width=True)
        filename = convert_data_to_hdf5(tg2, random.getrandbits(128))
        with open(filename, "rb") as f:
            st.download_button("Download generated data", file_name=filename, data=f, mime='application/x-hdf5')
    with Bent_real_col:
        st.write("### Bent real:")
        bent_real_ind = st.slider("Bent real index", 0, len(thresh_real_images[3])-1,0)
        bent_real_img = tr3[bent_real_ind].cpu().detach().numpy()
        st.image(bent_real_img, use_column_width=True)
    with Bent_gen_col:
        st.write("### Bent gen:")
        bent_gen_ind = st.slider("Bent gen index", 0, len(thresh_gen_images[3])-1,0)
        bent_gen_img = tg3[bent_gen_ind].cpu().detach().numpy()
        st.image(bent_gen_img, use_column_width=True)
        filename = convert_data_to_hdf5(tg3, random.getrandbits(128))
        with open(filename, "rb") as f:
            st.download_button("Download generated data", file_name=filename, data=f, mime='application/x-hdf5')

    st.write("For generated images:")
    st.write("Copyright 2022 Florian Griese "
             "Permission is hereby granted, free of charge, "
             "to any person obtaining a copy of this software and associated documentation files "
             "(the \"Software\"), " \
                            "to deal in the Software without restriction, including without limitation the rights to " \
                            "use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies " \
                            "of the Software, and to permit persons to whom the Software is furnished to do so, " \
                            "subject to the following conditions: The above copyright notice and this permission " \
                            "notice shall be included in all copies or substantial portions of the Software. " \
"THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, " \
                                "INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, " \
                                "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. " \
                                "IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, " \
                                "DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, " \
                                "ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR " \
                                "OTHER DEALINGS IN THE SOFTWARE.")
    print("app finished")
