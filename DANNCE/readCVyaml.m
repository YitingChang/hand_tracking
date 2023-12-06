%% Create a class
classdef readCVyaml
    % readCVyaml contains functions to read an openCV YAML file. 
    % reference: https://www.mathworks.com/help/vision/ref/cameraintrinsicsfromopencv.html
    methods(Static)

        function cameraParams = helperReadYAML(filename)
        % helperReadYAML reads an openCV YAML, filename, and returns a structure
        % with these fields: intrinsicMatrix,  distortionCoefficients, R, T.
        % These fileds are stored in the YAML file colon separated from their
        % values in different lines. 
        
            f = fopen(filename,'r');

            while ~feof(f)
        
                [name,value,isEmptyLine] = readCVyaml.helperReadYAMLLine(f);
                if isEmptyLine
                    continue
                end
                
                if strcmp(value,'!!opencv-matrix')
                    % If a value == '!!opencv-matrix' in openCV YAML files, it indicates a matrix in
                    % upcoming lines. Read the matrix from the upcoming lines.
                    value = readCVyaml.helperReadYAMLMatrix(f);
                end
         
                % Store post-processed value.
                cameraParams.(name) = value;
            end
            
            fclose(f);
        end
        
        function matrix = helperReadYAMLMatrix(f)
        %   helperReadYAMLMatrix reads a matrix from the openCV YAML file. A matrix in
        %   a openCV YAML file has four fields: rows, columns, dt and data. rows and col
        %   describe the matrix size. data is a continguous array of the matrix
        %   elements in row-major order. This helper function assumes the presence
        %   of all three fields of a matrix to return the correct matrix.
        
            numRows = 0;
            numCols = 0;
            data = [];
            data_sub = [];
            % Read numRows, numCols, dt and matrix data.
            while ~feof(f)
                [name,value,isEmptyLine] = readCVyaml.helperReadYAMLLine(f);
        
                if isEmptyLine
                    continue
                end
        
                switch name
                    case 'rows'
                        numRows = str2num(value); %#ok
                    case 'cols'
                        numCols = str2num(value); %#ok
                    case 'data'
                        data    = str2num(value); %#ok
                        if isempty(data) && isempty(data_sub) % data is in two lines
                            data_sub = value;
                        elseif isempty(data) && ~isempty(data_sub)
                            data = str2num([data_sub, value]);
                            break
                        end

                        % Terminate the while loop as data is the last 
                        % field of a matrix in the openCV YAML file.
                        if ~isempty(data)
                            break
                        end
                        
                    case 'dt'
                        continue
                    otherwise
                        % Terminate the while loop if any other field is
                        % encountered.
                        break
                end
            end
        
            if ~isempty(data) && numel(data) == numRows*numCols
                % Reshape the matrix using row-major order.
                matrix = reshape(data,[numCols numRows])';
            end
        end
        
        function [name,value,isEmptyLine] = helperReadYAMLLine(f)
        % The helperReadYAMLLine function reads a line of an openCV YAML file.
            
            % Read line from file.
            line = fgetl(f); 
        
            % Trim leading and trailing whitespaces.
            line = strtrim(line);
        
            if isempty(line) || line(1)=='#' || line(1) == '%' || line(1) == '-'
                % Empty line, comment, line starts with % (etc %YAML), or - (etc ---).
                name = '';
                value = '';
                isEmptyLine = true;
            elseif any(contains(line,':'))
                % Split the line to get name and value.
                c = strsplit(line,':');
                assert(length(c)==2,'Unexpected file format')
        
                name = c{1};
                value = strtrim(c{2}); % Trim leading whitespace.
                isEmptyLine = false;
            else % data is more than one line. 
                name = 'data';
                value = line;
                isEmptyLine = false;
            end
        end

    end
end
