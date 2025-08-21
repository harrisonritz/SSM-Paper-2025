function data = load_h5(fn)

info = h5info(fn);

data = struct;
for ii = 1:length(info.Datasets)

    name = info.Datasets(ii).Name;
    
    d = h5read(fn, ['/',name]);

    data.(name) = d;

end
