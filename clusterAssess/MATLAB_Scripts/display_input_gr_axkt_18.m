

function fighdl=display_input_gr_axkt_18(min_color, max_color, ny, nx, varargin)
% display a number of images in one window
% display_input_gr_axkt(min_color, max_color, ny, nx, color_map, images, image_titles)
%   min_color, max_color lowest and highest value values for each image displayed with the first and last color in color_map
%   if min_color==[], lowest and highest value  are calculated by mean(image)-+max_color*std(image) seperately for each image
%   if min_color==[] and max_color==[], lowest and highest value  are calculated min(image) and max(image) seperately for each image
%   ny   number of subplot rows
%   nx number of subplot columns
%   if any or both are empty [] they are calculated with respect to screen and image sizes
%   olor_map have the size n*3 otherwise it will be handled like an image
%   images: a list of images or a cellarray of images
%   image_titles a list of strings as title for each image: not required--> the image names or the subplot index is used as title
%
% Action: zoom at one image --> zoom all images
%           click at any point delivers for each image the value(s)
%           moving the crosshair by click or arrow keys
%           clicking key ''p''  or ''P'' select point interactively by coordinates
%           clicking key ''c''  or ''C'' change color of the crosshair
%           clicking key ''t''  or ''T'' change image display to image text
%           clicking key ''f''  or ''F'' change formats for text subwindow
%           clicking key ''g''  or ''G'' get a profile of the image
%           clicking key ''h''  or ''H'' show help for display_input_gr_axkt
%           clicking key ''l''  or ''L'' draw a polygon for labeling
%           clicking key ''r''  or ''R'' draw a polygon for relabeling of
%           the last one
%           clicking key ''a''  or ''A'' draw a polygon for approving of
%           the forelast
%           clicking key ''s''  or ''S'' save the polygon mask
%
% known bugs:   zooming at one subwindow allows only unzooming at the same subwindow
%               deactivate zoomin, zoomout and pan-button before positioning action
%Exapmles
% if 0
%     street1=imread('street1.jpg'); % Load image data
%     % street2=imread('street2.jpg');
%     street1_g = double(rgb2gray(street1));
%
%     % Darstellung mehrerer Bilder
%     % help display_input_gr_axkt
%     display_input_gr_axkt([], [], [], [], street1, street1_g);
%     % now: zoom or set zoom off and click or press key 'p' or key 'c' or the arrow keys
%     % unzoom only available at the subwindow, where the zoom action is done before
%
%     display_input_gr_axkt([], [], [], [], street1, street1_g, 'street original', 'street gray');  % with own titles (not required)
%
%     % oder
%     streets = {street1, street1_g};
%     names = {'street original', 'street gray'};
%     display_input_gr_axkt([], [], [], [], streets{:}, names{:});  % with own titles (not required)
%
% end
global pc_start inc_x inc_y click_inc_abs xpo ypo u v markerColor showText te te_visible ftext textfont textsize textxp textyp w_rad mask% h ax
ftext = '% 7.2f';
textfont = 'Arial';
textsize = 12;
textxp = -1;
textyp = -1;
w_rad = 1;
% pc_start = 1;
pc = [];
showText = 0;
% te = [];
own_color = 0;
ax = [];
inc_x = 0;
inc_y = 0;
markerColor =[1 0 0];
switch nargin
    case {0, 1, 2, 3, 4}
        fprintf(1,' usage: %s(min_color, max_color, ny, nx, images)\n   use for all images the same color range\n', mfilename);
        fprintf(1,' or usage: %s(min_color, max_color, ny, nx, map, images)\n   use for all images the colormap map\n', mfilename);
        fprintf(1,' or usage: %s([], fac, ny, nx, map, images)\n   use for each image [min_color max_color]=mean(image)±fac*std(image)\n', mfilename);
        fprintf(1,' or usage: %s([], [], ny, nx, map, images)\n   use for each image [min_color max_color]=[min(image) max(image)]\n', mfilename);
        fprintf(1,' or usage: %s([], [], [], [], map, images)\n  determine ny and nx based on the size of the first image and the number of the images\n', mfilename);
        fprintf(1,' or usage: %s(argumentlist without map)\n ', mfilename);
        fprintf(1,' or usage: %s(argumentlist list_of_images list_of_header) length of lists should be equal; header should be a string\n ', mfilename);
        fprintf(1,' or usage: %s(argumentlist list_of_pairs(image header)); header should be a string \n ', mfilename);
        fprintf(1,' Additional features: Cursor position by Click or arrows or press ''p'' for position input. \n ');
        fprintf(1,' press ''c'' for changing of the cross color. \n ');
        fprintf(1,' press ''t'' for the image as text. \n ');
        fprintf(1,' press ''f'' for formating the image as text. \n ');
        fprintf(1,' press ''g'' for getting a profile of the image. \n ');
        fprintf(1,' press ''l'' to label a polygon [Somewhere]. The variable mask will be created  \n ');
        fprintf(1,' press ''r'' to relabel  \n ');
        fprintf(1,' press ''a'' to approve  \n ');
        fprintf(1,' press ''s'' to save the mask  \n ');
        return;
        
    otherwise
        if size(min_color,1)==0 && size(max_color,1)==0  % use min and max of image
            own_color = 1;
        end
        if (size(varargin{1},2) == 3)  % colormap
            startim = 1;
            map = varargin{1};
        else
            startim = 0;
            map = gray(256);
        end
end;

nin = nargin-4-startim;
varargin_i = cell(nin);
varargin_h = cell(nin);
% check for header
ni = 0; % number of images
nih = 0;    % number of headers
for i=1: nin
    if isnumeric(varargin{i+startim}) || islogical(varargin{i+startim})
        ni = ni+1;
        varargin_i{ni}=varargin{i+startim};
    elseif ischar(varargin{i+startim})
        nih = nih+1;
        varargin_h{nih}=varargin{i+startim};
    end
end

% check if number of header == zero or equal number of images
if nih>0 && nih~=ni
    fprintf(1,'In %s number of images does not correspond with the number of headers. \n', mfilename);
    return;
end

% default header if no number of headers are given
if nih==0
    for i=1: ni
        head = inputname(i+4+startim);
        if strcmp(head,'')
            head = num2str(i);
        end
        varargin_h{i}=head;
    end
end

if ni==0    % no image
    fprintf(1,' usage: %s(min_color, max_color, ny, nx, map, images)\n', mfilename);
    return;
    
end
if size(min_color,1)==0 && size(max_color,1)~=0
    flag = 1;
    fac = max_color;
else
    flag = 0;
end;

% automatically determination of ny and nx
si_im = size(varargin_i{ni});
gm=get(0,'MonitorPosition');
if size(gm,1)>1
    gmxy = (gm(2,3)-gm(2,1)+1)/(gm(2,4)-gm(2,2));
else
    gmxy = (gm(1,3)-gm(1,1)+1)/(gm(1,4)-gm(1,2));
end;
if isempty(ny)
    if isempty(nx)
        nx = round(sqrt(gmxy*ni*si_im(1)/si_im(2)));
        nx=max(nx,1);
    end
    ny = ceil(ni/nx);
else
    if isempty(nx)
        nx = ceil(ni/ny);
    end
end
if (nx-1)*ny>=ni, nx=nx-1; end;
te = zeros(ny*nx,1);
te_visible = false(ny*nx,1);
%
for i=1:length(te)
    try
        set(te(i),'Visible','Off');
    catch
        %u = te(i);
        addprop(handle(te(i)), 'Visible');
        %set(u,'Visible','Off');
        set(te(i),'Visible','Off');
    end
end

%
fighdl=figure;
% model = @CursorPressFcn;
set(fighdl, 'WindowButtonDownFcn', @KeyPressFcn);
set(fighdl, 'KeyPressFcn', @KeyPressFcn);


dx = 0.04;  % nicht kleiner wählen sonst gibt es Ärger/ evtl. überschreibt ein subplot einen vorhergehenden
dy = dx;
Bx = (1-dx*(nx+1))/nx;
By = (1-dy*(ny+1))/ny;
for i = 1:min(nx*ny, ni)
    if i>=21
        disp i
    end
    k = mod(i-1,nx)+1;
    xp = mod(dx*k+(k-1)*Bx, 1-dx);
    yp = 1 - ((dy+By)*(1+floor((i-1)/nx)));
    %         ax(i) = subplot(ny, nx, i);
    ax(i) = subplot(ny, nx, i,'align');
    %     ninfi = find(-inf<varargin_i{i} && varargin_i{i}<inf);
    ninfi = (-inf<varargin_i{i} & varargin_i{i}<inf);
    tv = varargin_i{i}(ninfi);
    if flag==1
        me = mean(tv);
        sd = std(tv);
        min_color = me - fac*sd;
        max_color = me + fac*sd;
    end
    if own_color==1
        min_color = min(tv);
        max_color = max(tv);
    end
    if min_color==max_color
        max_color = min_color+1;
    end
    
    Properties(varargin_i{i}, min_color, max_color, map);
    
    %     title(ax(i),num2str(i));
    head = varargin_h{i};
    
    title(ax(i),head,'Interpreter','none','Units','normalized','Position',[0.5 1]);
    %         title(ax(i),inputname(i+4),'Interpreter','none','Position',[0.5*size(varargin{i+4},2) 0]);
    hold(ax(i),'on');
    %cb=colorbar;
    %     gcbp=get(cb,'OuterPosition');
    % %     gcbp(3) = 0.06;%0.5*gcbp(3);
    %     set(cb,'OuterPosition',gcbp);
end;

linkaxes(ax,'xy');
abg{1}=ax;
abg{2}=pc;
abg{3}=te;
abg{4}=te_visible;
set(gcf,'UserData',abg)
axhdls  = abg{1};
% hold on;

%__________________________________________________________________________
%______
%-----------------------------------------------------------------------
%-----------------------------------------------------------------------
%%%
%%%  Sub-function - CursorPressFcn
%%%
%-----------------------------------------------------------------------

%     function CursorPressFcn (src, evnt)
%
%         disp('CursorPressFcn')
%         evnt
%         inc_x = 0;
%         inc_y = 0;
%         click_inc_abs = 1
%         %         evnt
%         PressFcn (src, evnt);
%     end % function CursorPressFcn (src, evnt)
%-----------------------------------------------------------------------
%%%
%%%  Sub-function - KeyPressFcn
%%%
%-----------------------------------------------------------------------

    function KeyPressFcn (src, evnt)
        %         disp('KeyPressFcn')
        %         src, evnt
        click_inc_abs = 1;
        %         if isempty(evnt) || ~isfield(evnt,'Key') %|| strcmp(evnt.EventName, 'WindowMousePress') %isprop (evnt, 'Source')
        if isempty(evnt) || ~isprop(evnt,'Key') %|| strcmp(evnt.EventName, 'WindowMousePress') %isprop (evnt, 'Source')
            
            
            inc_x = 0;
            inc_y = 0;
            click_inc_abs = 1;
            %         evnt
            PressFcn ();%src, evnt);
        else
            click_inc_abs = 2;
            
            inc_x=0; inc_y=0;
            switch evnt.Key
                case 'uparrow'
                    inc_y = -1;
                case 'downarrow'
                    inc_y = +1;
                case 'leftarrow'
                    inc_x = -1;
                case 'rightarrow'
                    inc_x = +1;
                case {'p','P'}
                    prompt = {'Enter x:','Enter y:'};
                    dlg_title = 'Input for position';
                    num_lines = 1;
                    def = {num2str(u),num2str(v)};
                    answer = inputdlg(prompt,dlg_title,num_lines,def);
                    click_inc_abs = 3;
                    try
                        if isempty(answer)
                            xpo = u;
                            ypo = v;
                        else
                            xpo = str2double(answer{1});
                            ypo = str2double(answer{2});
                        end
                    catch
                        disp('Input is not a number.')
                    end
                    
                case {'f','F'}
                    prompt = {'Format text like %7.2f', 'Font', 'Size', 'X-Position', 'Y-Position', 'window_radius'};
                    dlg_title = 'Format for text';
                    num_lines = 1;
                    def = {ftext, textfont, num2str(textsize), num2str(textxp), num2str(textyp), num2str(w_rad)};
                    answer = inputdlg(prompt,dlg_title,num_lines,def);
                    click_inc_abs = 3;
                    try
                        if ~isempty(answer)
                            ftext = answer{1};
                            textfont = answer{2};
                            textsize = str2num(answer{3});
                            textxp = str2num(answer{4});
                            textyp = str2num(answer{5});
                            w_rad =  str2num(answer{6});
                            
                        end
                    catch
                        disp('Input is wrong.')
                    end
                    
                case {'c','C'}
                    markerColor = uisetcolor(pc(1), 'DialogTitle');
                case {'g','G'}
                    disp('Select at least 2 points. Stop with double click.');
                    improfile;
                    grid on
                case {'t','T'}
                    showText = get(gcbf,'CurrentAxes');
                    for i=1:size(axhdls,2)   % toggle on off for text
                        if axhdls(i)==showText  % only for the actual image
                            if te_visible(i)
                                set(te(i),'visible','off');
                                te_visible(i)=false;
                            else
                                set(te(i),'visible','on');
                                te_visible(i)=true;
                            end
                        end
                    end
                case {'l','L'}
                    if(sum(size(mask) - size(varargin_i{1}(:,:,1))) ~= 0)
                        mask = zeros(size(varargin_i{1}(:,:,1)));
                    end
                    hR = impoly();
                    polyXY =  hR.getPosition;
                    
                    BW_poly = poly2mask(polyXY(:,1),polyXY(:,2),size(mask,1),size(mask,2));
                    
                    input = inputdlg('enter label number:', 'Label');
                    label = str2num(input{:});
                    %                     label = 1;
                    
                    mask(BW_poly) =  label;
                    
                case {'r','R'}
                    if(sum(size(mask) - size(varargin_i{1}(:,:,1))) ~= 0)
                        mask = zeros(size(varargin_i{1}(:,:,1)));
                    end
                    
                    axCurPt = get(axhdls, {'CurrentPoint'});
                    pt = axCurPt{1};
                    u  = round(pt(1, 1));
                    v  = round(pt(1, 2));
                    img = varargin {end};
                    wxh = size(img);
                   % a = impixel();
                    if u > 0 && u <= wxh(2) && v > 0 && v <= wxh(1) && numel(wxh)== 2
                        BW_poly = img == img(v, u);
                        if sum(BW_poly(:)) > numel(BW_poly)/10
                            % das bedeutet wohl: verklickt, es wird kein
                            % relabeling stattfinden
                            disp('****************************************')
                            disp('*                                      *')
                            disp('*         Fuck! Fuck! Fuck!            *')
                            disp('*                                      *')
                            disp('****************************************')
                            return
                        end
                        %input = inputdlg('enter label number:', 'Label');
                        input = newid('enter label number:', 'Label');
                        label = str2num(input{:});
                        
                        %                     label = 1;
                        
                        mask(BW_poly) =  label;
                    end
                    
                case {'s','S'}
                    if(sum(size(mask) - size(varargin_i{1}(:,:,1))) == 0)
                        [FileName,PathName] = uiputfile('*.mat','save mask as');
                        save([PathName,FileName],'mask');
                        display_input_gr_axkt_18([], [], 1, 1, mask);
                        mask = [];
                    else
                        disp('no mask found')
                    end
                case {'h','H'}
                    doc display_input_gr_axkt
                    %                 otherwise
                    disp(evnt.Key);
                    
            end
            %                 evnt
            %         return;
            PressFcn (src, evnt);
        end
    end % function KeyPressFcn (src, evnt)

%-----------------------------------------------
    function PressFcn (src, evnt)
        
        %         disp('PressFcn')
        %         evnt, src
        %         global pc h ax
        %         src, evnt
        fighdl  = gcbf;
        %         pos0    = get (fighdl, 'Position');
        abg=get(fighdl,'UserData');
        axhdls  = findobj (fighdl, 'type', 'axes');
        %axhdls  = abg{1};
        pc=abg{2};
        te=abg{3};
        te_visible = abg{4};
        
        oldU    = get(axhdls, {'Units'});
        set(axhdls, 'Units', 'pixels');
        %         axPos   = get([axhdls], {'Position'});
        axCurPt = get(axhdls, {'CurrentPoint'});
        set(axhdls, {'Units'}, oldU);
        %         XLim = get(axhdls, 'XLim');
        %         YLim = get(axhdls, 'YLim');
        % fprintf('aha\n');
        for k = 1 : length(axhdls)
            %     fprintf('aha %d\n',k);
            axa  = axhdls(k);
            XLim = get (axa, 'XLim');
            YLim = get (axa, 'YLim');
            click_inc_abs;
            switch click_inc_abs
                case 1
                    %                     axCurPt = get([axhdls], {'CurrentPoint'});
                    pt = axCurPt{k};
                    x  = pt(1, 1);  xpo=x;
                    y  = pt(1, 2);  ypo=y;
                case 2
                    %                     pt = axCurPt{k};
                    x  = u+inc_x;
                    y  = v+inc_y;
                    
                case 3
                    x = xpo;
                    y = ypo;
            end
            %             set(axa,'CurrentPoint',[x y 0])
            if x >= XLim(1) & x <= XLim(2) & y >= YLim(1) & y <= YLim(2)
                imhdl = findobj (axa, 'type', 'image');
                img = get(imhdl, 'CData');
                wxh = size(img);
                u = round(x);
                v = round(y);
                if u > 0 && u <= wxh(2) && v > 0 && v <= wxh(1),
                    
                    
                    for i = 1 : numel(axhdls) %min(numel(axhdls), numel(te_visible))
                        imhdl = findobj (axhdls(i), 'type', 'image');
                        img = get(imhdl(end), 'CData');
                        wxh = size(img);
                        if u > 0 && u <= wxh(2) && v > 0 && v <= wxh(1)
                            %                             if pc_start==1
                            if length(pc)<i || (length(pc)>=i && pc(i)==0)
                                hold(axhdls(i),'on');
                                pc(i)=plot(axhdls(i),u,v,'+r','LineWidth',2,'MarkerSize',10, 'MarkerEdgeColor', markerColor);
                            else
                                set(pc(i),'XData',u,'YData',v, 'MarkerEdgeColor',markerColor);
                            end
                            
                            
                            if size(img,3)==1
                                xlabel(axhdls(i),sprintf('g(%d,%d)=%g',u,v,img(v,u)),'Visible','on','Units','normalized','Position',[.5 0]);
                            else
                                %                             xlabel(axhdls(i),sprintf('g(%d,%d)=(%d,%d,%d)',u,v,img(v,u,1),img(v,u,2),img(v,u,3)),...
                                %                             xlabel(axhdls(i),sprintf('g(%d,%d)=(%.2f,%.2f,%.2f)',u,v,img(v,u,1),img(v,u,2),img(v,u,3)),...
                                xlabel(axhdls(i),sprintf('g(%d,%d)=(%g,%g,%g)',u,v,img(v,u,1),img(v,u,2),img(v,u,3)),...
                                    'Visible','on','Units','normalized','Position',[.5 0]);
                            end;
                            %                             if axhdls(i)==showText
                            
                            if te_visible(i)
                                %                                 w_rad = 1;
                                sel = img(max(1,v-w_rad):min(wxh(1),v+w_rad), max(1,u-w_rad):min(wxh(2),u+w_rad));
                                if size(img,3)==3
                                    sel = permute(img(max(1,v-w_rad):min(wxh(1),v+w_rad), max(1,u-w_rad):min(wxh(2),u+w_rad),:),[1 3 2]);
                                    try
                                        nr_sel=num2str(sel,['(' ftext ftext ftext ') ' ]);     %
                                    catch
                                        disp('Format for numbers is wrong.')
                                    end
                                    %                                     nr_sel=num2str(sel,ftext);
                                else
                                    try
                                        nr_sel=num2str(sel,ftext);
                                    catch
                                        disp('Format for numbers is wrong.')
                                    end
                                end
                                if te(i)>0
                                    set(te(i),'string',nr_sel,'Fontname',textfont, 'fontsize', textsize, 'Position',[textxp textyp 0]);
                                else
                                    if textxp<0
                                        textxp = wxh(2)*0.5;
                                    end
                                    if textyp<0
                                        textyp = wxh(1)*0.5;
                                    end
                                    te(i)=text(textxp, textyp, nr_sel, 'HorizontalAlignment','center','VerticalAlignment','middle', ...
                                        'BackgroundColor',[1 1 1],'Fontname',textfont, 'fontsize', textsize);
                                end
                            else
                                % Hinzugefügt!!!!
                                try
                                    set(te(i),'Visible','Off');
                                catch
                                    addprop(handle(te(i)), 'Visible');
                                    set(te(i),'Visible','Off');
                                end
                                
                                %set(te(i),'visible','off');
                            end     % te_visible(i)
                            
                            %                             end
                        end
                    end
                    %                     pc_start = 0;
                    % plot(axhdls(3-k),u,v,'+r');
                end % if u > 0 & u <= wxh(2) & v > 0 & v <= wxh(1),
                break;
            end % if x >= XLim(1) & x <= XLim(2) & y >= YLim(1) & y <= YLim(2)
            
        end % for k = 1 : length(axhdls)
        abg{1}=ax;
        abg{2}=pc;
        abg{3}=te;
        set(fighdl,'UserData',abg);
        drawnow
    end % function PressFcn (src, evnt)

end % display_input_gr_ax


