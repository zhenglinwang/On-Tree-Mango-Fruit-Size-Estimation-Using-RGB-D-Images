function [BW] = filterStalk(imBinary, mean_end)

% remove stalks
%mean_end = 8;
[hh, ww] = size(imBinary);
for lines = 1:hh
    line_pixel = imBinary(lines,:);
    line_pixel(end) = 0;
    indx_zeros = find(line_pixel==0);

    last_zero_pos = 0;
    for i=1:1:length(indx_zeros)
        current_zero_pos = double(indx_zeros(i));
        if current_zero_pos-last_zero_pos <= mean_end
            imBinary(lines, last_zero_pos+1:current_zero_pos-1) = 0;
        end

        last_zero_pos = current_zero_pos;
    end
end

for lines = 1:ww
    line_pixel = imBinary(:, lines);
    line_pixel(end) = 0;
    indx_zeros = find(line_pixel==0);

    last_zero_pos = 0;
    for i=1:1:length(indx_zeros)
        current_zero_pos = double(indx_zeros(i));
        if current_zero_pos-last_zero_pos <= mean_end
            imBinary(last_zero_pos+1:current_zero_pos-1, lines) = 0;
        end

        last_zero_pos = current_zero_pos;
    end
end
BW = imBinary;