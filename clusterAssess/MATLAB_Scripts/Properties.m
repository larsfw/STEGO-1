function Properties(z_raster, min, max, map)

switch nargin
    case 0
        disp 'Properties needs at least image parameter';
        return;
    case 1
        min=0;
        max=255;
        map=gray;
    case 2
        max=255;
        map=gray;
    case 3
        map = gray;
    case 4
        
    otherwise
        disp 'Usage: Properties(z_raster, min, max, map)';
        return;
end;
        
 h=imagesc(z_raster);
 set(h, 'CDataMapping','scaled')
 colormap(map)
 axis image off
 set(gca,'TickDir','out');
 if min < max
     caxis([ double(min) double(max)]);
 end

    

return;