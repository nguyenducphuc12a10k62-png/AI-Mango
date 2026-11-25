"""Mango AI feature extraction v7.0 (classical CV / ML only).

Drop-in replacement for MangoAI_starter/features.py. Same public functions,
extract() return tuple. Grading features change; re-extract and retrain.
Ripeness keeps the original v2 color pipeline.
No label or held-out test data is used by this module.
"""
import cv2
import numpy as np
from PIL import Image, ImageOps
cv2.setNumThreads(1)
FEATURE_VERSION = "7.1"
# Fraction of the fruit bounding box short side, not fraction of area.
QUALITY_RIM_FRACTION = 0.10
QUALITY_CORE_FRACTION = 0.10

def load_image(path):
    with Image.open(path) as im: return cv2.cvtColor(np.asarray(ImageOps.exif_transpose(im).convert('RGB')),cv2.COLOR_RGB2BGR)

def resize(img, limit=384):
    h,w=img.shape[:2]; scale=min(1.,limit/max(h,w)); return cv2.resize(img,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)

def _ellipse(size):
    size = max(3, int(size) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _largest_filled(binary):
    binary = np.ascontiguousarray((binary > 0).astype(np.uint8) * 255)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('No usable foreground')
    contour = max(contours, key=cv2.contourArea)
    result = np.zeros_like(binary)
    cv2.drawContours(result, [contour], -1, 255, cv2.FILLED)
    return result, contour


def _border_pixels(img, width):
    return np.concatenate((img[:width].reshape(-1, img.shape[-1]),
                           img[-width:].reshape(-1, img.shape[-1]),
                           img[:, :width].reshape(-1, img.shape[-1]),
                           img[:, -width:].reshape(-1, img.shape[-1])), axis=0)


def _segment_grabcut(img):
    """GrabCut + guarded neutral-background refinement + contour cleanup.

    The caller may crop an ROI first (app.py already supports this). Keep a
    little background around the entire fruit. A colorful/complex background
    is not removed by color threshold alone.
    """
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError('Expected a BGR image')
    img = resize(img)
    h, w = img.shape[:2]
    if min(h, w) < 60:
        raise ValueError('Image too small')
    sm = cv2.GaussianBlur(img, (3, 3), 0)
    margin = max(2, round(min(h, w) * .018))
    gc = np.zeros((h, w), np.uint8)
    rect = (margin, margin, w - 2 * margin, h - 2 * margin)
    cv2.setRNGSeed(42)
    cv2.grabCut(sm, gc, rect, np.zeros((1, 65), np.float64),
                np.zeros((1, 65), np.float64), 4, cv2.GC_INIT_WITH_RECT)
    mask, contour = _largest_filled((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD))
    _, _, bw, bh = cv2.boundingRect(contour)
    short = min(bw, bh)
    # Only use this extra prior on approximately neutral, low-chroma borders.
    # It trims attached grey/white background and grey cast shadows, not dark
    # patches inside the fruit (the chromatic body silhouette is hole-filled).
    lab = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab = lab[:, :, 1:]
    border = _border_pixels(lab, max(2, round(min(h, w) * .035)))[:, 1:]
    bg = np.median(border, axis=0)
    bg_dist = np.linalg.norm(border - bg, axis=1)
    chroma_bg = np.linalg.norm(border - 128., axis=1)
    dist = np.linalg.norm(ab - bg, axis=2)
    if np.percentile(chroma_bg, 85) < 23 and np.percentile(bg_dist, 85) < 13:
        cutoff = max(14., float(np.percentile(bg_dist, 95)) + 6.)
        chromatic = ((dist > cutoff) & (mask > 0)).astype(np.uint8) * 255
        chromatic = cv2.morphologyEx(chromatic, cv2.MORPH_CLOSE, _ellipse(short * .035))
        try:
            support, support_contour = _largest_filled(chromatic)
            fraction = np.count_nonzero((support > 0) & (mask > 0)) / max(1, np.count_nonzero(mask))
            if fraction >= .70:
                # Small tolerance for low-chroma skin and genuine rim lesions.
                # Restore shallow inward notches at dark rim lesions, but
                # do not add any pixels beyond the original GrabCut mask.
                hull = cv2.convexHull(support_contour)
                if cv2.contourArea(hull) <= 1.08 * max(1., cv2.contourArea(support_contour)):
                    cv2.drawContours(support, [hull], -1, 255, cv2.FILLED)
                support = cv2.dilate(support, _ellipse(short * .025))
                refined = cv2.bitwise_and(mask, support)
                mask, contour = _largest_filled(refined)
        except ValueError:
            pass  # If color evidence is poor, retain the GrabCut result.
    # Smooth tiny protrusions (e.g. stem) while keeping the full fruit body.
    _, _, bw, bh = cv2.boundingRect(contour)
    k = _ellipse(min(bw, bh) * .075)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if np.count_nonzero(opened) >= .92 * np.count_nonzero(mask):
        mask, contour = _largest_filled(opened)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(3))
    mask, contour = _largest_filled(mask)
    fraction = np.count_nonzero(mask) / mask.size
    if not .07 <= fraction <= .93:
        raise ValueError(f'Invalid foreground area ({fraction:.1%}); adjust ROI')
    x, y, bw, bh = cv2.boundingRect(contour)
    if min(bw, bh) < 35:
        raise ValueError('Fruit region too small')
    return img, mask


def auto_fruit_box(img):
    """Return a padded xyxy box and filled color-support mask at input scale.
    Intended for single green/yellow mango on a neutral backdrop; reject
    ambiguous candidates instead of treating a colored picture frame as fruit.
    """
    h,w=img.shape[:2]
    sm=cv2.GaussianBlur(img,(5,5),0)
    H,S,V=cv2.split(cv2.cvtColor(sm,cv2.COLOR_BGR2HSV))
    L,A,B=cv2.split(cv2.cvtColor(sm,cv2.COLOR_BGR2LAB))
    color=((H>=8)&(H<=105)&(S>=42)&(V>=35)&(B>=140)).astype(np.uint8)*255
    k=_ellipse(max(3,min(h,w)*.015))
    color=cv2.morphologyEx(color,cv2.MORPH_OPEN,k)
    color=cv2.morphologyEx(color,cv2.MORPH_CLOSE,k)
    n,ids,stats,_=cv2.connectedComponentsWithStats(color,8)
    candidates=[]
    for j in range(1,n):
        x,y,bw,bh,area=stats[j]
        # Components touching the picture border are often frames or background.
        if x<=1 or y<=1 or x+bw>=w-1 or y+bh>=h-1: continue
        if area<.045*h*w or min(bw,bh)<30 or area>.85*h*w: continue
        support,c=_largest_filled(ids==j)
        hull=cv2.convexHull(c)
        ca=cv2.contourArea(c); ha=cv2.contourArea(hull)
        fill=ca/max(1,bw*bh)
        if not .30<=fill<=.88 or ca/max(1,ha)<.77: continue
        if ha<=1.14*ca: cv2.drawContours(support,[hull],-1,255,cv2.FILLED)
        candidates.append((area,support,(x,y,bw,bh)))
    if not candidates: return None
    candidates.sort(key=lambda q:q[0],reverse=True)
    if len(candidates)>1 and candidates[1][0]>.65*candidates[0][0]: return None
    _,support,(x,y,bw,bh)=candidates[0]
    pad=max(8,round(.14*min(bw,bh)))
    return (max(0,x-pad),max(0,y-pad),min(w,x+bw+pad),min(h,y+bh+pad)),support


def _check_mask(mask):
    _,c=_largest_filled(mask)
    x,y,w,h=cv2.boundingRect(c)
    fill=cv2.contourArea(c)/max(1,w*h)
    if fill>.90:
        raise ValueError('Foreground looks rectangular; adjust ROI or use a neutral background')


def segment(img):
    """Automatic ROI, then GrabCut. Keep the legacy result if color evidence is weak."""
    if img is None or img.ndim!=3 or img.shape[2]!=3: raise ValueError('Expected a BGR image')
    img=resize(img)
    found=auto_fruit_box(img)
    # Keep the established segmentation when it already agrees with the fruit.
    old=None
    try:
        old=_segment_grabcut(img)
        _check_mask(old[1])
    except (ValueError,cv2.error): old=None
    if found is None:
        if old is not None: return old
        raise ValueError('Cannot locate a reliable mango region; adjust ROI')
    (x0,y0,x1,y1),support=found
    if old is not None:
        m=old[1]>0; c=support>0
        overlap=np.count_nonzero(m&c)/max(1,np.count_nonzero(c))
        ratio=np.count_nonzero(m)/max(1,np.count_nonzero(c))
        if overlap>=.85 and .82<=ratio<=1.30: return old
    crop=img[y0:y1,x0:x1].copy(); prior=support[y0:y1,x0:x1]
    try:
        cropped,mask=_segment_grabcut(crop)
        _check_mask(mask)
        a=mask>0; b=prior>0
        overlap=np.count_nonzero(a&b)/max(1,np.count_nonzero(b))
        ratio=np.count_nonzero(a)/max(1,np.count_nonzero(b))
        if overlap>=.84 and .80<=ratio<=1.30: return cropped,mask
    except (ValueError,cv2.error): pass
    # Color support is only a localization cue, not a substitute for a verified
    # lesion-preserving segmentation. Do not silently classify an invalid mask.
    if old is not None and np.count_nonzero((old[1]>0)&(support>0))/max(1,np.count_nonzero(support))>=.85 and np.count_nonzero(old[1])<=1.45*np.count_nonzero(support): return old
    raise ValueError('Automatic ROI found but fruit mask is unreliable; adjust ROI')


def _masked_gaussian(channel, weight, sigma):
    """Local smooth reference using fruit pixels only, never black background."""
    numerator = cv2.GaussianBlur(channel * weight, (0, 0), sigma)
    denominator = cv2.GaussianBlur(weight, (0, 0), sigma)
    return numerator / np.maximum(denominator, 1e-6)


def _defect_mask(lab, mask):
    """Dark/brown candidates: local lightness AND chromatic anomalies.

    Avoid absolute darkness as a stand-alone rule. Two spatial scales help
    distinguish gradual shading from actual spots. No class label is used.
    """
    L, A, B = cv2.split(lab.astype(np.float32))
    area = int(np.count_nonzero(mask))
    x, y, bw, bh = cv2.boundingRect(mask)
    short = min(bw, bh)
    # Remove only a narrow uncertain rim, scaled to the fruit, not a wide band.
    inner = cv2.erode(mask, _ellipse(short * .018)) > 0
    weight = (mask > 0).astype(np.float32)
    small = max(2., short * .035)
    large = max(5., short * .105)
    Ls = _masked_gaussian(L, weight, small)
    Ll = _masked_gaussian(L, weight, large)
    As = _masked_gaussian(A, weight, small)
    Al = _masked_gaussian(A, weight, large)
    Bs = _masked_gaussian(B, weight, small)
    Bl = _masked_gaussian(B, weight, large)
    dl_small = Ls - L
    dl_large = Ll - L
    # Brown/dark lesions often shift toward red and/or lose yellow chroma.
    # A shadow mainly changes L; chromatic checks reduce false positives.
    da = np.maximum(A - As, A - Al)
    db = np.maximum(Bs - B, Bl - B)
    contrast = np.maximum(dl_small, dl_large)
    chromatic = (da > 5.) | (db > 8.) | ((da > 3.) & (db > 4.))
    seed = inner & (((contrast > 13.) & chromatic) |
                    ((dl_small > 23.) & (dl_large > 19.)))
    # Keep nearby milder pixels only if connected to a high-confidence seed.
    weak_color = (da > 2.) | (db > 3.)
    grow = inner & (((contrast > 7.) & weak_color) | (dl_small > 14.))
    grow |= seed
    grow = (grow.astype(np.uint8) * 255)
    grow = cv2.morphologyEx(grow, cv2.MORPH_CLOSE, _ellipse(3))
    grow[~inner] = 0
    num, ids, stats, _ = cv2.connectedComponentsWithStats(grow, connectivity=8)
    minimum = max(4, round(area * .00025))
    defect = np.zeros_like(mask)
    sizes = []
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # Detect long, diffuse shadow ribbons rather than deleting a whole rim.
    # Compact lesions near the edge are intentionally allowed to survive.
    for j in range(1, num):
        size = int(stats[j, cv2.CC_STAT_AREA])
        if size < minimum:
            continue
        region = ids == j
        if np.count_nonzero(seed & region) < max(2, round(size * .07)):
            continue
        edge_fraction = float(np.mean(distance[region] < .07 * short))
        length = max(stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT])
        contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        circularity = 4. * np.pi * size / max(perimeter * perimeter, 1.)
        diffuse_ribbon = (length > .38 * short and edge_fraction > .32 and
                          circularity < .23 and float(np.median(dl_small[region])) < 10.)
        if diffuse_ribbon:
            # A true dark spot can be connected to a shadow. Preserve only
            # sharply localized dark cores in this otherwise diffuse region.
            cores = region & (((dl_small > 20.) & chromatic) | (dl_small > 28.))
            ncore, core_ids, core_stats, _ = cv2.connectedComponentsWithStats(cores.astype(np.uint8), connectivity=8)
            for q in range(1, ncore):
                if core_stats[q, cv2.CC_STAT_AREA] >= minimum:
                    defect[core_ids == q] = 255
        else:
            defect[region] = 255
    # Count the actual retained components after ribbon suppression.
    n, ids, stats, _ = cv2.connectedComponentsWithStats(defect, connectivity=8)
    for j in range(1, n):
        size = int(stats[j, cv2.CC_STAT_AREA])
        if size >= minimum:
            sizes.append(size)
        else:
            defect[ids == j] = 0
    return defect, sizes, inner, contrast


def _color_features(img, mask, defect, inner):
    area = int(np.count_nonzero(mask))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    H, S, V = cv2.split(hsv)
    L, A, B = cv2.split(lab)
    # Keep the v2 color masking and hue definitions unchanged.
    valid = inner & (defect == 0) & (S >= 25) & (V >= 35) & (V <= 250)
    if valid.sum() < max(50, .15 * area):
        raise ValueError('Too few usable color pixels; adjust ROI or retake image')
    colors=H[valid]; f={
      'green_ratio':float(np.mean((colors>=32)&(colors<=95))), 'yellow_ratio':float(np.mean((colors>=20)&(colors<32))),
      'orange_ratio':float(np.mean((colors>=5)&(colors<20))),
      'hue_sin':float(np.sin(colors*np.pi/90).mean()), 'hue_cos':float(np.cos(colors*np.pi/90).mean()),
      'sat_mean':float(S[valid].mean()), 'lab_a_mean':float(A[valid].mean()), 'lab_b_mean':float(B[valid].mean()),
      'rgb_r':float(img[:,:,2][valid].mean()), 'rgb_g':float(img[:,:,1][valid].mean()), 'rgb_b':float(img[:,:,0][valid].mean())}
    # K-means on Lab chroma: assign clusters to semantic hue bins, not cluster IDs.
    ab=lab[valid][:,1:].astype(np.float32); rng=np.random.default_rng(42)
    idx=rng.choice(len(ab),min(1800,len(ab)),replace=False); sample=ab[idx]; hh=colors[idx]
    for n in (3,5,7):
        cv2.setRNGSeed(42)
        _,labels,_=cv2.kmeans(sample,n,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,35,.15),2,cv2.KMEANS_PP_CENTERS)
        labels=labels.ravel(); ratios={'green':0.,'yellow':0.,'orange':0.}
        for j in range(n):
            vals=hh[labels==j]
            if not len(vals): continue
            hue=float(np.median(vals)); name='green' if 32<=hue<=95 else 'yellow' if 20<=hue<32 else 'orange' if 5<=hue<20 else None
            if name: ratios[name]+=len(vals)/len(hh)
        for name,v in ratios.items(): f[f'km{n}_{name}']=v
    return f

# Grading: normalized geometric + local photometric features (no raw global
# lightness or raw global chroma standard deviation).
DEFECT = ['defect_ratio','largest_defect_ratio','defect_count_per_10k','mean_defect_ratio']
ROBUST = ['strong_defect_ratio','largest_strong_ratio','relative_dark_p95',
          'relative_dark_tail','local_chroma_p95','dark_residual_mad']
GRADE = DEFECT + ROBUST
DEFECT_MASK = DEFECT + ROBUST[:2]
DEFECT_EXPERT = DEFECT_MASK + ['defect_concentration']
LOCAL = ['defect_color_contrast','defect_texture_contrast','scratch_ratio']
DEFECT_LOCAL = DEFECT_EXPERT + LOCAL
BROWN = ['brown_defect_ratio','largest_brown_ratio','brown_color_p90']
GRADE_ALL = list(dict.fromkeys(GRADE + DEFECT_LOCAL + BROWN))
HSV=['green_ratio','yellow_ratio','orange_ratio','hue_sin','hue_cos','sat_mean']
LAB=['lab_a_mean','lab_b_mean']
RGB=['rgb_r','rgb_g','rgb_b']


def normalize_fruit(img, mask, target_short=180):
    """Scale the fruit itself, not the entire photo. Preserve aspect ratio."""
    x,y,w,h = cv2.boundingRect(mask)
    if min(w,h)<35: raise ValueError('Fruit region too small')
    scale = min(target_short/min(w,h), 440/max(w,h))
    pad = max(3,round(min(w,h)*.04))
    x0,y0=max(0,x-pad),max(0,y-pad)
    x1,y1=min(img.shape[1],x+w+pad),min(img.shape[0],y+h+pad)
    a=img[y0:y1,x0:x1]; m=mask[y0:y1,x0:x1]
    dims=(max(1,round(a.shape[1]*scale)),max(1,round(a.shape[0]*scale)))
    a=cv2.resize(a,dims,interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    m=cv2.resize(m,dims,interpolation=cv2.INTER_NEAREST)
    return a,m,(x0,y0,x1,y1)


def _robust_reference(channel, weight, sigma, lightness=False):
    """Robust local illumination estimate, downweight dark outliers.
    Masked smoothing avoids mixing the artificial black background with skin.
    """
    base=_masked_gaussian(channel,weight,sigma)
    if lightness:
        delta=np.maximum(base-channel,0.)
        w=weight/(1.+(delta/12.)**2)
        base=_masked_gaussian(channel,w,sigma)
    return base


def quality_interior(mask):
    """Remove narrow stem/protrusions and ignore an unreliable peripheral band.

    Operates at normalized fruit resolution. Does not modify the ripeness mask.
    The outer 10% of the short side is excluded from grading.
    """
    m = np.ascontiguousarray((mask > 0).astype(np.uint8) * 255)
    _, _, w, h = cv2.boundingRect(m)
    short = min(w, h)
    if short < 35:
        raise ValueError('Fruit region too small for quality analysis')
    # Opening removes narrow protrusions such as stems, not interior spots.
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, _ellipse(short * .12),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0)
    if np.count_nonzero(opened) >= .92 * np.count_nonzero(m):
        body, _ = _largest_filled(opened)
        body = cv2.bitwise_and(body, m)
    else:
        body = m.copy()
    # Pad with background so distance remains correct if the fruit touches ROI.
    padded = cv2.copyMakeBorder(body, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    inner = (body > 0) & (dist >= max(2., short * QUALITY_RIM_FRACTION))
    if inner.sum() < 100 or inner.sum() < .5 * np.count_nonzero(m):
        raise ValueError('Insufficient reliable fruit interior; adjust ROI')
    return body, inner, dist, short


def grading_features(img,mask):
    a,m,box=normalize_fruit(img,mask)
    lab=cv2.cvtColor(a,cv2.COLOR_BGR2LAB).astype(np.float32)
    L,A,B=cv2.split(lab)
    body,inner,dist,short=quality_interior(m)
    # Single central ROI; the excluded rim is never scored.
    # References use the body to avoid an artificial ROI boundary.
    central = inner.copy()
    if central.sum() < 100: raise ValueError("Insufficient central skin")
    weight=(body>0).astype(np.float32)
    area=int(inner.sum())
    # Relative lightness deficit rather than absolute L. The reference follows
    # slow illumination variation; two scales retain small and broad lesions.
    ss=max(2.,short*.035); sl=max(5.,short*.12)
    ls=_robust_reference(L,weight,ss,True); ll=_robust_reference(L,weight,sl,True)
    rs=(ls-L)/np.maximum(ls,45.); rl=(ll-L)/np.maximum(ll,45.)
    dark=np.maximum(rs,rl)
    da=np.maximum(A-_masked_gaussian(A,weight,ss),A-_masked_gaussian(A,weight,sl))
    db=np.maximum(_masked_gaussian(B,weight,ss)-B,_masked_gaussian(B,weight,sl)-B)
    # Ripe yellow -> natural green patches are not brown: avoid green shifts
    # (negative a); use local loss of yellow and/or red shift + dark evidence.
    chromatic=(da>5.) | ((db>8.) & (da>-2.)) | ((da>2.5)&(db>4.))
    # Mask-aware multi-scale black top-hat: outside the fruit is excluded
    # from both dilation and erosion so artificial black borders cannot seed defects.
    def blackhat(k):
        kernel=_ellipse(k)
        dil=cv2.dilate(np.where(weight>0,L,-1e4).astype(np.float32),kernel)
        close=cv2.erode(np.where(weight>0,dil,1e4).astype(np.float32),kernel)
        return np.maximum(close-L,0.)
    bh_small=blackhat(short*.055)
    bh_large=blackhat(short*.16)
    # Fine-scale texture is supplementary, not mandatory for a smooth bruise.
    smooth=_masked_gaussian(L,weight,1.15)
    texture=_masked_gaussian(np.abs(L-smooth),weight,2.)
    texture_ref=_masked_gaussian(texture,weight,max(5.,short*.09))
    texture_delta=np.maximum(texture-texture_ref,0.)
    localized=(bh_small>8.) | (bh_large>15.)
    sharp_core=(bh_small>22.) & (rs>.17) & (texture_delta>1.1)
    seed=inner & (((dark>.085)&chromatic&localized) | sharp_core)
    strong=inner & (((dark>.17)&chromatic&(bh_large>19.)) | ((bh_small>29.)&(rs>.23)&(texture_delta>1.1)))
    weak_color=(da>1.5)|((db>3.)&(da>-2.))
    weak=inner & (((dark>.045)&weak_color&(bh_large>7.)) | ((bh_small>12.)&(rs>.10)&(texture_delta>.6)))
    weak |= seed
    grow=cv2.morphologyEx(weak.astype(np.uint8)*255,cv2.MORPH_CLOSE,_ellipse(3))
    grow[~inner]=0
    n,ids,stats,_=cv2.connectedComponentsWithStats(grow,connectivity=8)
    minimum=max(5,round(area*.00030))
    defect=np.zeros_like(m)
    for j in range(1,n):
        size=int(stats[j,cv2.CC_STAT_AREA]);
        if size<minimum: continue
        r=ids==j
        if (seed&r).sum()<max(2,round(size*.08)): continue
        # All candidates already lie inside the central ROI.
        if not np.any(r&seed): continue
        defect[r]=255
    n,ids,stats,_=cv2.connectedComponentsWithStats(defect,connectivity=8); sizes=[]
    for j in range(1,n):
        size=int(stats[j,cv2.CC_STAT_AREA])
        if size<minimum: defect[ids==j]=0
        else: sizes.append(size)
    core=(strong&(defect>0)).astype(np.uint8)
    n,_,stats,_=cv2.connectedComponentsWithStats(core,connectivity=8)
    core_sizes=stats[1:,cv2.CC_STAT_AREA] if n>1 else np.array([0])
    # Local relative deficits and robust tails replace raw global L/a/b std.
    dr=np.maximum(dark[central],0.)
    q95=float(np.percentile(dr,95)); tail=dr[dr>=q95]
    ch=np.sqrt(np.maximum(da[central],0.)**2+np.maximum(db[central],0.)**2)
    f={'defect_ratio':sum(sizes)/area,'largest_defect_ratio':max(sizes,default=0)/area,
       'defect_count_per_10k':len(sizes)*10000/area,'mean_defect_ratio':float(np.mean(sizes)/area) if sizes else 0.,
       'strong_defect_ratio':float(core.sum()/area),'largest_strong_ratio':float(max(core_sizes)/area),
       'relative_dark_p95':q95,'relative_dark_tail':float(np.mean(tail)),
       'local_chroma_p95':float(np.percentile(ch,95)),
       'dark_residual_mad':float(np.median(np.abs(dr-np.median(dr))))}
    # Fraction of retained defect area in its largest connected component.
    f['defect_concentration']=float(max(sizes,default=0)/sum(sizes)) if sizes else 0.
    chromacontrast=np.sqrt(np.maximum(da,0.)**2+np.maximum(db,0.)**2)
    valid_def=defect>0
    f['defect_color_contrast']=float(np.median(chromacontrast[valid_def])) if valid_def.any() else 0.
    f['defect_texture_contrast']=float(np.median(texture_delta[valid_def])) if valid_def.any() else 0.
    # Elongated components are a proxy for scratches, not a diagnosis.
    n,ids,stats,_=cv2.connectedComponentsWithStats(defect,connectivity=8)
    scratch=0
    for j in range(1,n):
        yy,xx=np.where(ids==j)
        if len(xx)<minimum: continue
        eig=np.linalg.eigvalsh(np.cov(np.stack([xx,yy])))
        elongation=np.sqrt(max(eig[-1],0.)/max(eig[0],.25))
        if elongation>=3. and max(stats[j,cv2.CC_STAT_WIDTH],stats[j,cv2.CC_STAT_HEIGHT])>=short*.06:
            scratch+=len(xx)
    f['scratch_ratio']=float(scratch/area)
    # Separate stronger brown evidence from generic dark/texture candidates.
    # A redward local a* shift or local loss of yellow is evidence, not a proof.
    brown = valid_def & (dark>.085) & chromatic
    n,_,bstats,_=cv2.connectedComponentsWithStats(brown.astype(np.uint8),connectivity=8)
    f['brown_defect_ratio']=float(brown.sum()/area)
    f['largest_brown_ratio']=float(max(bstats[1:,cv2.CC_STAT_AREA],default=0)/area)
    f['brown_color_p90']=float(np.percentile(chromacontrast[brown],90)) if brown.any() else 0.

    # Diagnostic only: not part of the model feature lists.
    f['quality_roi_ratio']=float(area/max(1,np.count_nonzero(m)))
    f['quality_core_ratio']=float(central.sum()/max(1,np.count_nonzero(m)))
    f['peripheral_defect_ratio']=float(np.count_nonzero((defect>0)&~central)/area)
    # Map defect overlay to the original segmentation resolution for the UI.
    x0,y0,x1,y1=box; overlay=np.zeros_like(mask)
    overlay[y0:y1,x0:x1]=cv2.resize(defect,(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST)
    valid_roi=np.zeros_like(mask)
    valid_roi[y0:y1,x0:x1]=cv2.resize(inner.astype(np.uint8),(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST)
    overlay[(mask==0)|(valid_roi==0)]=0
    return f,overlay


def extract(img):
    img,mask=segment(img)
    # The v2 color exclusion mask is intentionally separate: photometric
    # corrections for Grading must not alter the hue used for Ripeness.
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    color_defect,_,inner,_=_defect_mask(lab,mask)
    f=_color_features(img,mask,color_defect,inner)
    grade,defect=grading_features(img,mask)
    f.update(grade)
    return f,img,mask,defect
