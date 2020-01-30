# Pre-processing and Augmentation

# Read Images - Actual Image and Table Mask Image
actual_img = cv2.imread(actual_img_path)
masked_img = cv2.imread(masked_img_path)
# Resize Images
actual_img = cv2.resize(actual_img,(512,512),interpolation=cv2.INTER_AREA)
masked_img = cv2.resize(masked_img,(512,512),interpolation=cv2.INTER_NEAREST)
rows,cols,c1 = actual_img.shape
rows1,cols1,c2 = masked_img.shape
lista = [1,2,3,4]
for u in range(random.choice(lista)):
    for k in np.random.normal(2,0.4,size=4):
        for j in np.random.normal(0,10,4):
            answer2 = random.choice(['yes', 'no'])
            answer3 = random.choice(['yes', 'no'])
            answer4 = random.choice(['yes', 'no'])
            if answer2=='yes':
                gauss = np.random.normal(0,0.01,actual_img.size)
                gauss = gauss.reshape(actual_img.shape[0],actual_img.shape[1],actual_img.shape[2]).astype('uint8')
                actual_img = actual_img + actual_img * gauss
            if answer3=='yes':
                vals = len(np.unique(actual_img))
                vals = 2 ** np.ceil(np.log2(vals))
                actual_img = np.random.poisson(np.abs(actual_img * vals)) / float(np.abs(vals))
            if answer4=='yes':
                noise_img = random_noise(actual_img, mode='salt',amount=0.12)
                actual_img = np.array(255*noise_img, dtype = 'uint8')
            listb = [3,4,5]
            b = random.choice(listb)
            M = cv2.getRotationMatrix2D((cols,rows),j,k)
            actual_img = cv2.warpAffine(actual_img,M,(cols,rows), borderMode=cv2.BORDER_REFLECT)
            masked_img = cv2.warpAffine(masked_img,M,(cols1,rows1), borderMode=cv2.BORDER_REFLECT)
