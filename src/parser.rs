use std::fmt;
use std::num::{ParseFloatError, ParseIntError};
use std::path::PathBuf;
use types::*;

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub struct Pos {
    r: usize,
    c: usize,
}

impl fmt::Display for Pos {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.r, self.c)
    }
}

impl Pos {
    fn new(r: usize, c: usize) -> Self {
        Pos { r, c }
    }
}

#[derive(Debug, PartialEq, Clone)]
struct Token {
    pos: Pos,
    ty: TokenType,
}

impl Token {
    fn remove_pos(self) -> TokenType {
        self.ty
    }
}

#[derive(Debug, PartialEq)]
pub enum PreDirective {
    // Transforms
    Identity(Pos),
    Translate(Pos, Vector3f),
    Scale(Pos, Vector3f),
    Rotate(Pos, f64, Vector3f),
    LookAt(Pos, Matrix3f),
    CoordinateSystem(Pos, String),
    CoordSysTransform(Pos, String),
    Transform(Pos, Matrix3f),
    ConcatTransform(Pos, Matrix3f),
    // Transform timing
    ActiveTransform(Pos, String),
    TransformTimes(Pos, f64, f64),

    // Scene-wide options
    Camera(DirectiveStruct),
    Sampler(DirectiveStruct),
    Film(DirectiveStruct),
    Filter(DirectiveStruct),
    Integrator(DirectiveStruct),
    Accelerator(DirectiveStruct),

    Include(Pos, PathBuf),
}

#[derive(Debug, PartialEq)]
pub struct ParamSet {
    pub bools: Vec<Param<bool>>,
    pub ints: Vec<Param<isize>>,
    pub floats: Vec<Param<f64>>,
    pub point2fs: Vec<Param<Point2f>>,
    pub vector2fs: Vec<Param<Vector2f>>,
    pub point3fs: Vec<Param<Point3f>>,
    pub vector3fs: Vec<Param<Vector3f>>,
    pub normal3fs: Vec<Param<Normal3f>>,
    pub spectra: Vec<Param<Spectrum>>,
    pub strings: Vec<Param<String>>,
    pub textures: Vec<Param<String>>,
}

impl Default for ParamSet {
    fn default() -> Self {
        ParamSet {
            bools: vec![],
            ints: vec![],
            floats: vec![],
            point2fs: vec![],
            vector2fs: vec![],
            point3fs: vec![],
            vector3fs: vec![],
            normal3fs: vec![],
            spectra: vec![],
            strings: vec![],
            textures: vec![],
        }
    }
}

impl ParamSet {
    #[cfg(test)]
    fn add_floats(mut self, p: &[Param<f64>]) -> Self {
        {
            let this = &mut self;
            this.floats.extend_from_slice(p);
        }
        self
    }

    #[cfg(test)]
    fn add_ints(mut self, p: &[Param<isize>]) -> Self {
        {
            let this = &mut self;
            this.ints.extend_from_slice(p);
        }
        self
    }

    #[cfg(test)]
    fn add_strings(mut self, p: &[Param<String>]) -> Self {
        {
            let this = &mut self;
            this.strings.extend_from_slice(p);
        }
        self
    }

    #[cfg(test)]
    fn add_spectra(mut self, p: &[Param<Spectrum>]) -> Self {
        {
            let this = &mut self;
            this.spectra.extend_from_slice(p);
        }
        self
    }

    #[cfg(test)]
    fn add_textures(mut self, p: &[Param<String>]) -> Self {
        {
            let this = &mut self;
            this.textures.extend_from_slice(p);
        }
        self
    }
}

#[derive(Debug, PartialEq)]
pub enum Directive {
    Material(DirectiveStruct),
    Shape(DirectiveStruct),
    LightSource(DirectiveStruct),
    AreaLightSource(DirectiveStruct),
    Attribute(BlockStruct),
    Transform(BlockStruct),
    World(BlockStruct),
    Include(Pos, PathBuf),
    Texture(TextureStruct),
}

#[derive(Debug, PartialEq)]
pub struct DirectiveStruct {
    ty: String,
    pos: Pos,
    param_set: ParamSet,
}

#[derive(Debug, PartialEq)]
pub struct TextureStruct {
    name: String,
    ty: String,
    class: String,
    pos: Pos,
    param_set: ParamSet,
}

#[derive(Debug, PartialEq)]
pub struct BlockStruct {
    pos: Pos,
    children: Vec<Directive>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Param<T> {
    name: String,
    pos: Pos,
    values: Vec<T>,
}

impl<T> Param<T> {
    fn new(name: &str, pos: Pos, values: Vec<T>) -> Self {
        Param {
            name: name.to_owned(),
            pos,
            values,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
enum TokenType {
    Identifier(String),
    Int(isize),
    Float(f64),
    Str(String),
    LBracket,
    RBracket,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Tokenizer {
    chars: Vec<char>,
    index: usize,
    pos: Pos,
    prev_pos: Pos,
}

pub struct Parser {
    tokens: Vec<Token>,
    index: usize,
}

pub type TokenizerResult<T> = Result<T, TokenizerError>;
pub type ParserResult<T> = Result<T, ParserError>;

#[derive(Debug, PartialEq)]
pub enum TokenizerError {
    Str(String),
    Eof,
}

#[derive(Debug, PartialEq)]
pub enum ParserError {
    Str(String),
    Eof,
}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ParserError::Str(s) => write!(f, "{}", s),
            ParserError::Eof => write!(f, "EOF"),
        }
    }
}

impl From<ParseIntError> for TokenizerError {
    fn from(err: ParseIntError) -> Self {
        TokenizerError::Str(err.to_string())
    }
}

impl From<ParseFloatError> for TokenizerError {
    fn from(err: ParseFloatError) -> Self {
        TokenizerError::Str(err.to_string())
    }
}

impl From<TokenizerError> for ParserError {
    fn from(err: TokenizerError) -> Self {
        match err {
            TokenizerError::Str(s) => ParserError::Str(s),
            TokenizerError::Eof => ParserError::Eof,
        }
    }
}

impl Tokenizer {
    fn new(input: &str) -> Self {
        Tokenizer {
            chars: input.replace("\r\n", "\n").chars().collect(),
            index: 0,
            pos: Pos::new(1, 1),
            prev_pos: Pos::new(1, 1),
        }
    }

    fn tokenize(input: &str) -> TokenizerResult<Vec<Token>> {
        let mut tokenizer = Tokenizer::new(input);
        let mut tokens = vec![];
        while !tokenizer.is_empty() {
            match tokenizer.tokenize_one() {
                Ok(tok) => tokens.push(tok),
                Err(TokenizerError::Eof) => break,
                Err(err) => return Err(err),
            }
        }
        Ok(tokens)
    }

    fn peek(&self) -> TokenizerResult<char> {
        if !self.is_empty() {
            Ok(self.chars[self.index])
        } else {
            Err(TokenizerError::Eof)
        }
    }

    fn next(&mut self) -> TokenizerResult<char> {
        if self.is_empty() {
            return Err(TokenizerError::Eof);
        }
        let ch = self.chars[self.index];
        self.prev_pos = self.pos;
        self.pos = if ch == '\n' {
            Pos::new(self.pos.r + 1, 1)
        } else {
            Pos::new(self.pos.r, self.pos.c + 1)
        };
        self.index += 1;
        Ok(ch)
    }

    fn is_empty(&self) -> bool {
        self.index >= self.chars.len()
    }

    fn tokenize_one(&mut self) -> TokenizerResult<Token> {
        match self.peek() {
            Ok('#') => {
                while let Ok(ch) = self.next() {
                    if ch == '\n' {
                        break;
                    }
                }
                self.tokenize_one()
            }
            Ok('"') => self.tokenize_str(),
            Ok('[') => {
                let tok = Ok(Token {
                    pos: self.pos,
                    ty: TokenType::LBracket,
                });
                let _ = self.next();
                tok
            }
            Ok(']') => {
                let tok = Ok(Token {
                    pos: self.pos,
                    ty: TokenType::RBracket,
                });
                let _ = self.next();
                tok
            }
            Ok(c) if c == '-' || c == '+' || c == '.' || c.is_numeric() => self.tokenize_num(),
            Ok(c) if c.is_whitespace() => {
                while let Ok(ch) = self.peek() {
                    match ch {
                        ch if ch.is_whitespace() => {
                            self.next()?;
                        }
                        _ => break,
                    }
                }
                self.tokenize_one()
            }
            Ok(c) if c.is_alphabetic() => self.tokenize_id(),
            Ok(c) => Err(TokenizerError::Str(format!(
                "{} tokenize_one(): unexpected char '{}'",
                self.pos, c,
            ))),
            Err(err) => Err(err),
        }
    }

    fn tokenize_num(&mut self) -> TokenizerResult<Token> {
        let start_pos = self.pos;
        let mut num_chars = vec![];
        match self.peek() {
            Ok(ch @ '-') => {
                num_chars.push(ch);
                let _ = self.next();
            }
            Ok('+') => {
                let _ = self.next();
            }
            _ => (),
        }
        let mut is_float = false;
        loop {
            match self.peek() {
                Ok(ch) if ch.is_numeric() => num_chars.push(self.next()?),
                Ok(ch @ '.') => if is_float {
                    return Err(TokenizerError::Str(format!(
                        "{} tokenize_num(): unexpected char '{}'",
                        self.pos, ch
                    )));
                } else {
                    is_float = true;
                    num_chars.push(self.next()?);
                },
                Ok('e') => {
                    if num_chars.is_empty() {
                        return Err(TokenizerError::Str(format!(
                            "{} tokenize_num(): no number",
                            self.pos
                        )));
                    }
                    let _ = self.next();
                    match self.tokenize_num()?.remove_pos() {
                        TokenType::Float(f) => {
                            return Ok(Token {
                                pos: start_pos,
                                ty: TokenType::Float(
                                    num_chars.into_iter().collect::<String>().parse::<f64>()?
                                        * f64::powf(10.0, f),
                                ),
                            })
                        }
                        TokenType::Int(i) if i >= 0 && !is_float => {
                            return Ok(Token {
                                pos: start_pos,
                                ty: TokenType::Int(
                                    num_chars.into_iter().collect::<String>().parse::<isize>()?
                                        * isize::pow(10, i as u32),
                                ),
                            })
                        }
                        TokenType::Int(i) => {
                            return Ok(Token {
                                pos: start_pos,
                                ty: TokenType::Float(
                                    num_chars.into_iter().collect::<String>().parse::<f64>()?
                                        * f64::powf(10.0, i as f64),
                                ),
                            })
                        }
                        _ => {
                            return Err(TokenizerError::Str(format!(
                                "{} tokenize_num(): no number",
                                start_pos
                            )))
                        }
                    }
                }
                Ok(_) | Err(TokenizerError::Eof) => break,
                Err(err) => return Err(err),
            }
        }
        if num_chars.is_empty() {
            return Err(TokenizerError::Str(format!(
                "{} tokenize_num(): no number",
                self.pos
            )));
        }
        if is_float {
            Ok(Token {
                pos: start_pos,
                ty: TokenType::Float(num_chars.into_iter().collect::<String>().parse()?),
            })
        } else {
            Ok(Token {
                pos: start_pos,
                ty: TokenType::Int(num_chars.into_iter().collect::<String>().parse()?),
            })
        }
    }

    fn tokenize_str(&mut self) -> TokenizerResult<Token> {
        let start_pos = self.pos;
        // Skip over opening quote
        match self.next() {
            Ok(_) => (),
            Err(TokenizerError::Eof) => {
                return Err(TokenizerError::Str(format!(
                    "{} tokenize_str(): EOF while processing string",
                    self.pos
                )))
            }
            Err(err) => return Err(err),
        }
        let mut str_chars = vec![];
        loop {
            match self.next() {
                Ok('\\') => match self.next() {
                    Ok('n') => str_chars.push('\n'),
                    Ok(ch) => str_chars.push(ch),
                    Err(TokenizerError::Eof) => {
                        return Err(TokenizerError::Str(format!(
                            "{} tokenize_str(): EOF while processing escape seq",
                            self.prev_pos
                        )))
                    }
                    Err(err) => return Err(err),
                },
                Ok('"') => {
                    return Ok(Token {
                        pos: start_pos,
                        ty: TokenType::Str(str_chars.into_iter().collect()),
                    });
                }
                Ok(ch) => str_chars.push(ch),
                Err(TokenizerError::Eof) => {
                    return Err(TokenizerError::Str(format!(
                        "{} tokenize_str(): EOF while processing string",
                        self.pos
                    )))
                }
                Err(err) => return Err(err),
            }
        }
    }

    fn tokenize_id(&mut self) -> TokenizerResult<Token> {
        let start_pos = self.pos;
        let mut id_chars = vec![self.next()?];
        loop {
            match self.peek() {
                Ok(ch) if ch.is_alphanumeric() => id_chars.push(self.next()?),
                Ok(_) | Err(TokenizerError::Eof) => break,
                Err(err) => return Err(err),
            }
        }
        if id_chars.is_empty() {
            return Err(TokenizerError::Str(format!(
                "{} tokenize_id(): no identifier",
                self.pos
            )));
        }
        Ok(Token {
            pos: start_pos,
            ty: TokenType::Identifier(id_chars.into_iter().collect()),
        })
    }
}

impl Parser {
    fn new(input: &str) -> ParserResult<Self> {
        Ok(Parser {
            tokens: Tokenizer::tokenize(input)?,
            index: 0,
        })
    }

    fn peek(&self) -> ParserResult<Token> {
        if !self.is_empty() {
            Ok(self.tokens[self.index].clone())
        } else {
            Err(ParserError::Eof)
        }
    }

    fn next(&mut self) -> ParserResult<Token> {
        if self.is_empty() {
            return Err(ParserError::Eof);
        }
        let tok = self.tokens[self.index].clone();
        self.index += 1;
        Ok(tok)
    }

    fn is_empty(&self) -> bool {
        self.index >= self.tokens.len()
    }

    fn pos(&self) -> ParserResult<Pos> {
        if self.index >= self.tokens.len() {
            Err(ParserError::Eof)
        } else {
            Ok(self.tokens[self.index].pos)
        }
    }

    fn set_index(&mut self, index: usize) {
        self.index = index;
    }

    fn parse_param_lists(&mut self, param_set: &mut ParamSet) -> ParserResult<()> {
        while self.parse_param_list(param_set).is_ok() {}
        Ok(())
    }

    fn parse_param_list(&mut self, param_set: &mut ParamSet) -> ParserResult<()> {
        match self.peek() {
            Ok(Token {
                pos,
                ty: TokenType::Str(s),
            }) => {
                let split_s: Vec<&str> = s.split_whitespace().collect();
                if split_s.len() != 2 {
                    return Err(ParserError::Str(format!(
                        "{} parse_param_list(): expecting a string with two arguments but got {}",
                        pos, s
                    )));
                }
                let ty = split_s[0];
                let var = split_s[1];
                match ty {
                    "integer" => {
                        self.next()?;
                        param_set.ints.push(Param::new(
                            var,
                            pos,
                            self.parse_one_or_list(Self::parse_int, "int")?,
                        ));
                    }
                    "float" => {
                        self.next()?;
                        param_set.floats.push(Param::new(
                            var,
                            pos,
                            self.parse_one_or_list(Self::parse_float, "float")?,
                        ));
                    }
                    "point2" => {
                        self.next()?;
                        param_set.point2fs.push(Param::new(
                            var,
                            pos,
                            self.parse_list(Self::parse_point2f, "point2f")?,
                        ));
                    }
                    "vector2" => {
                        self.next()?;
                        param_set.vector2fs.push(Param::new(
                            var,
                            pos,
                            self.parse_list(Self::parse_vector2f, "vector2f")?,
                        ));
                    }
                    "bool" => {
                        self.next()?;
                        param_set.bools.push(Param::new(
                            var,
                            pos,
                            self.parse_one_or_list(Self::parse_bool, "bool")?,
                        ));
                    }
                    "string" => {
                        self.next()?;
                        param_set.strings.push(Param::new(
                            var,
                            pos,
                            self.parse_one_or_list(Self::parse_string, "string")?,
                        ));
                    }
                    "rgb" => {
                        self.next()?;
                        let rgb = self.parse_list_of_n(Self::parse_float, 3, "float")?;
                        param_set.spectra.push(Param::new(
                            var,
                            pos,
                            vec![Spectrum::Rgb(rgb[0], rgb[1], rgb[2])],
                        ));
                    }
                    "xyz" => {
                        self.next()?;
                        let rgb = self.parse_list_of_n(Self::parse_float, 3, "float")?;
                        param_set.spectra.push(Param::new(
                            var,
                            pos,
                            vec![Spectrum::Xyz(rgb[0], rgb[1], rgb[2])],
                        ));
                    }
                    "spectrum" => {
                        self.next()?;
                        param_set
                            .spectra
                            .push(Param::new(var, pos, vec![self.parse_spectrum()?]));
                    }
                    "blackbody" => {
                        self.next()?;
                        let rgb = self.parse_list_of_n(Self::parse_float, 2, "float")?;
                        param_set.spectra.push(Param::new(
                            var,
                            pos,
                            vec![Spectrum::Blackbody(rgb[0], rgb[1])],
                        ));
                    }
                    "texture" => {
                        self.next()?;
                        param_set
                            .textures
                            .push(Param::new(var, pos, vec![self.parse_string()?]));
                    }
                    _ => {
                        return Err(ParserError::Str(format!(
                            "{} parse_param_list(): expecting a type but got {}",
                            pos, ty
                        )))
                    }
                }
                Ok(())
            }
            Ok(Token { pos, ty }) => Err(ParserError::Str(format!(
                "{} parse_param_list(): expected param list, got {:?}",
                pos, ty
            ))),
            Err(ParserError::Eof) => Err(ParserError::Str(
                "parse_param_list(): EOF while processing param list".to_owned(),
            )),
            Err(err) => Err(err),
        }
    }

    /// Returns error if list is empty
    fn parse_list<T>(
        &mut self,
        parser: fn(&mut Self) -> ParserResult<T>,
        name: &str,
    ) -> ParserResult<Vec<T>> {
        match self.peek() {
            Ok(Token {
                ty: TokenType::LBracket,
                pos: start_pos,
            }) => {
                let _ = self.next();
                let values = self.parse_many(parser, name)?;
                match self.peek() {
                    Ok(Token {
                        ty: TokenType::RBracket,
                        ..
                    }) => {
                        if values.is_empty() {
                            Err(ParserError::Str(format!(
                                "{} parse_list_{}(): parsed empty list",
                                start_pos, name,
                            )))
                        } else {
                            self.next()?;
                            Ok(values)
                        }
                    }
                    Ok(Token { pos, ty }) => Err(ParserError::Str(format!(
                        "{pos} parse_list_{name}(): expected ']' but got {ty:?}",
                        pos = pos,
                        name = name,
                        ty = ty
                    ))),
                    Err(ParserError::Eof) => Err(ParserError::Str(format!(
                        "parse_list_{}(): EOF while processing list",
                        name
                    ))),
                    Err(err) => Err(err),
                }
            }
            Ok(Token { pos, ty }) => Err(ParserError::Str(format!(
                "{pos} parse_list_{name}(): expected '[' but got {ty:?}",
                name = name,
                pos = pos,
                ty = ty
            ))),
            Err(ParserError::Eof) => Err(ParserError::Str(format!(
                "parse_list_{}(): EOF while processing list",
                name
            ))),
            Err(err) => Err(err),
        }
    }

    fn parse_list_of_n<T>(
        &mut self,
        parser: fn(&mut Self) -> ParserResult<T>,
        n: usize,
        name: &str,
    ) -> ParserResult<Vec<T>> {
        match self.peek() {
            Ok(Token {
                ty: TokenType::LBracket,
                pos: start_pos,
            }) => {
                let _ = self.next();
                let values = self.parse_n(parser, n, name)?;
                match self.peek() {
                    Ok(Token {
                        ty: TokenType::RBracket,
                        ..
                    }) => {
                        if values.is_empty() {
                            Err(ParserError::Str(format!(
                                "{pos} parse_list_of_{n}_{name}s(): parsed empty list",
                                name = name,
                                n = n,
                                pos = start_pos,
                            )))
                        } else {
                            self.next()?;
                            Ok(values)
                        }
                    }
                    Ok(Token { pos, ty }) => Err(ParserError::Str(format!(
                        "{pos} parse_list_of_{n}_{name}s(): expected ']' but got {ty:?}",
                        name = name,
                        n = n,
                        pos = pos,
                        ty = ty
                    ))),
                    Err(ParserError::Eof) => Err(ParserError::Str(format!(
                        "parse_list_of_n_{}s(): EOF while processing list",
                        name
                    ))),
                    Err(err) => Err(err),
                }
            }
            Ok(Token { pos, ty }) => Err(ParserError::Str(format!(
                "{pos} parse_list_of_n_{name}s(): expected '[' but got {ty:?}",
                name = name,
                pos = pos,
                ty = ty
            ))),
            Err(ParserError::Eof) => Err(ParserError::Str(format!(
                "parse_list_of_n_{}(): EOF while processing list",
                name
            ))),
            Err(err) => Err(err),
        }
    }

    fn parse_one_or_list<T>(
        &mut self,
        parser: fn(&mut Self) -> ParserResult<T>,
        name: &str,
    ) -> ParserResult<Vec<T>> {
        match self.peek() {
            Ok(Token {
                ty: TokenType::LBracket,
                ..
            }) => self.parse_list(parser, name),
            Ok(_) => Ok(vec![parser(self)?]),
            Err(ParserError::Eof) => Err(ParserError::Str(format!(
                "parse_one_or_list_{}(): EOF while processing list",
                name
            ))),
            Err(err) => Err(err),
        }
    }

    fn parse_int(&mut self) -> ParserResult<isize> {
        match self.peek() {
            Ok(Token {
                ty: TokenType::Int(i),
                ..
            }) => {
                let _ = self.next();
                Ok(i)
            }
            Ok(Token { pos, ty }) => Err(ParserError::Str(format!(
                "{} parse_int(): expected int but got {:?}",
                pos, ty
            ))),
            Err(ParserError::Eof) => Err(ParserError::Str(
                "parse_int(): EOF while parsing int".to_owned(),
            )),
            Err(err) => Err(err),
        }
    }

    fn parse_string(&mut self) -> ParserResult<String> {
        match self.peek() {
            Ok(Token {
                ty: TokenType::Str(s),
                ..
            }) => {
                let _ = self.next();
                Ok(s)
            }
            Ok(Token { pos, ty }) => Err(ParserError::Str(format!(
                "{} parse_string(): expected string but got {:?}",
                pos, ty
            ))),
            Err(ParserError::Eof) => Err(ParserError::Str(
                "parse_string(): EOF while processing string".to_owned(),
            )),
            Err(err) => Err(err),
        }
    }

    fn parse_identifier(&mut self) -> ParserResult<String> {
        match self.peek() {
            Ok(Token {
                ty: TokenType::Identifier(id),
                ..
            }) => {
                let _ = self.next();
                Ok(id)
            }
            Ok(Token { pos, ty }) => Err(ParserError::Str(format!(
                "{} parse_identifier(): expected identifier but got {:?}",
                pos, ty
            ))),
            Err(ParserError::Eof) => Err(ParserError::Str(
                "parse_identifier(): EOF while processing identifier".to_owned(),
            )),
            Err(err) => Err(err),
        }
    }

    fn parse_bool(&mut self) -> ParserResult<bool> {
        match self.peek() {
            Ok(Token {
                pos,
                ty: TokenType::Str(s),
            }) => match s.as_ref() {
                "true" => {
                    self.next()?;
                    Ok(true)
                }
                "false" => {
                    self.next()?;
                    Ok(false)
                }
                _ => Err(ParserError::Str(format!(
                    "{} parse_bool(): expected bool but got {:?}",
                    pos, s
                ))),
            },
            Ok(Token { pos, ty }) => Err(ParserError::Str(format!(
                "{} parse_bool(): expected int but got {:?}",
                pos, ty
            ))),
            Err(ParserError::Eof) => Err(ParserError::Str(
                "parse_bool(): EOF while parsing bool".to_owned(),
            )),
            Err(err) => Err(err),
        }
    }

    fn parse_float(&mut self) -> ParserResult<f64> {
        match self.peek() {
            Ok(Token {
                ty: TokenType::Float(f),
                ..
            }) => {
                self.next()?;
                Ok(f)
            }
            Ok(Token {
                ty: TokenType::Int(i),
                ..
            }) => {
                self.next()?;
                Ok(i as f64)
            }
            Ok(Token { pos, ty }) => Err(ParserError::Str(format!(
                "{} parse_float(): expected float but got {:?}",
                pos, ty
            ))),
            Err(ParserError::Eof) => Err(ParserError::Str(
                "parse_float(): EOF while parsing float".to_owned(),
            )),
            Err(err) => Err(err),
        }
    }

    fn parse_floats(&mut self) -> ParserResult<Vec<f64>> {
        self.parse_many(Self::parse_float, "float")
    }

    fn parse_many<T>(
        &mut self,
        parser: fn(&mut Self) -> ParserResult<T>,
        name: &str,
    ) -> ParserResult<Vec<T>> {
        let mut values = vec![];
        loop {
            match parser(self) {
                Ok(i) => values.push(i),
                Err(err) => {
                    if values.is_empty() {
                        return Err(ParserError::Str(format!(
                            "parse_many_{name}s(): error while processing {name}s:\n  {}",
                            err,
                            name = name,
                        )));
                    } else {
                        return Ok(values);
                    }
                }
            }
        }
    }

    fn parse_n<T>(
        &mut self,
        parser: fn(&mut Self) -> ParserResult<T>,
        n: usize,
        name: &str,
    ) -> ParserResult<Vec<T>> {
        assert!(n > 0);
        let start_index = self.index;
        let mut values = vec![];
        for _ in 0..n {
            match parser(self) {
                Ok(i) => values.push(i),
                Err(err) => {
                    self.set_index(start_index);
                    return Err(ParserError::Str(format!(
                        "parse_{n}_{name}s(): error while processing {name}s:\n  {err:?}",
                        err = err,
                        name = name,
                        n = n
                    )));
                }
            }
        }
        Ok(values)
    }

    fn parse_point2f(&mut self) -> ParserResult<Point2f> {
        let floats = self.parse_n(Self::parse_float, 2, "float")?;
        Ok(Point2f::new(floats[0], floats[1]))
    }

    fn parse_vector2f(&mut self) -> ParserResult<Vector2f> {
        let floats = self.parse_n(Self::parse_float, 2, "float")?;
        Ok(Vector2f::new(floats[0], floats[1]))
    }

    fn parse_directive(&mut self) -> ParserResult<Directive> {
        let start_pos = self.pos()?;
        let id = self.parse_identifier()?;
        let mut param_set = ParamSet::default();
        match id.as_ref() {
            "Material" => {
                let ty = self.parse_string()?;
                self.parse_param_lists(&mut param_set)?;
                Ok(Directive::Material(DirectiveStruct {
                    ty,
                    pos: start_pos,
                    param_set,
                }))
            }
            "Shape" => {
                let ty = self.parse_string()?;
                self.parse_param_lists(&mut param_set)?;
                Ok(Directive::Shape(DirectiveStruct {
                    ty,
                    pos: start_pos,
                    param_set,
                }))
            }
            "LightSource" => {
                let ty = self.parse_string()?;
                self.parse_param_lists(&mut param_set)?;
                Ok(Directive::LightSource(DirectiveStruct {
                    ty,
                    pos: start_pos,
                    param_set,
                }))
            }
            "AreaLightSource" => {
                let ty = self.parse_string()?;
                self.parse_param_lists(&mut param_set)?;
                Ok(Directive::AreaLightSource(DirectiveStruct {
                    ty,
                    pos: start_pos,
                    param_set,
                }))
            }
            "AttributeBegin" => Ok(Directive::Attribute(BlockStruct {
                pos: start_pos,
                children: self.parse_directives()?,
            })),
            "TransformBegin" => Ok(Directive::Transform(BlockStruct {
                pos: start_pos,
                children: self.parse_directives()?,
            })),
            "WorldBegin" => Ok(Directive::World(BlockStruct {
                pos: start_pos,
                children: self.parse_directives()?,
            })),
            "Include" => Ok(Directive::Include(
                start_pos,
                PathBuf::from(self.parse_string()?),
            )),
            "Texture" => {
                let name = self.parse_string()?;
                let ty = self.parse_string()?;
                let class = self.parse_string()?;
                self.parse_param_lists(&mut param_set)?;
                Ok(Directive::Texture(TextureStruct {
                    name,
                    ty,
                    class,
                    pos: start_pos,
                    param_set,
                }))
            }
            _ => Err(ParserError::Str(format!(
                "{} parse_directive(): unknown identifier {}",
                start_pos, id
            ))),
        }
    }

    fn parse_directives(&mut self) -> ParserResult<Vec<Directive>> {
        let mut directives = vec![];
        loop {
            match self.peek()? {
                Token {
                    ty: TokenType::Identifier(ref s),
                    ..
                } => {
                    if s == "AttributeEnd" || s == "TransformEnd" || s == "WorldEnd" {
                        return Ok(directives);
                    } else {
                        directives.push(self.parse_directive()?);
                    }
                }
                Token { pos, ty } => {
                    return Err(ParserError::Str(format!(
                        "{} parse_directives(): expected identifier or block close but got {:?}",
                        pos, ty
                    )));
                }
            }
        }
    }

    fn parse_predirective(&mut self) -> ParserResult<PreDirective> {
        let start_pos = self.pos()?;
        let mut param_set = ParamSet::default();
        match self.parse_identifier()?.as_ref() {
            "Identity" => Ok(PreDirective::Identity(start_pos)),
            "Translate" => {
                let floats = self.parse_floats()?;
                if floats.len() != 3 {
                    return Err(ParserError::Str(format!(
                        "{} parse_predirective(): Translate expects 3 floats but was given {}",
                        start_pos,
                        floats.len()
                    )));
                }
                Ok(PreDirective::Translate(
                    start_pos,
                    Vector3f::new(floats[0], floats[1], floats[2]),
                ))
            }
            "Scale" => {
                let floats = self.parse_floats()?;
                if floats.len() != 3 {
                    return Err(ParserError::Str(format!(
                        "{} parse_predirective(): Scale expects 3 floats but was given {}",
                        start_pos,
                        floats.len()
                    )));
                }
                Ok(PreDirective::Scale(
                    start_pos,
                    Vector3f::new(floats[0], floats[1], floats[2]),
                ))
            }
            "Rotate" => {
                let floats = self.parse_floats()?;
                if floats.len() != 4 {
                    return Err(ParserError::Str(format!(
                        "{} parse_predirective(): Rotate expects 4 floats but was given {}",
                        start_pos,
                        floats.len()
                    )));
                }
                Ok(PreDirective::Rotate(
                    start_pos,
                    floats[0],
                    Vector3f::new(floats[1], floats[2], floats[3]),
                ))
            }
            "LookAt" => {
                let floats = self.parse_floats()?;
                if floats.len() != 9 {
                    return Err(ParserError::Str(format!(
                        "{} parse_predirective(): LookAt expects 9 floats but was given {}",
                        start_pos,
                        floats.len()
                    )));
                }
                Ok(PreDirective::LookAt(
                    start_pos,
                    Matrix3f::new(
                        floats[0], floats[1], floats[2], floats[3], floats[4], floats[5],
                        floats[6], floats[7], floats[8],
                    ),
                ))
            }
            "CoordinateSystem" => {
                let name = self.parse_string()?;
                Ok(PreDirective::CoordinateSystem(start_pos, name))
            }
            "CoordSysTransform" => {
                let name = self.parse_string()?;
                Ok(PreDirective::CoordSysTransform(start_pos, name))
            }
            "Transform" => {
                let floats = self.parse_floats()?;
                if floats.len() != 9 {
                    return Err(ParserError::Str(format!(
                        "{} parse_predirective(): Transform expects 9 floats but was given {}",
                        start_pos,
                        floats.len()
                    )));
                }
                Ok(PreDirective::Transform(
                    start_pos,
                    Matrix3f::new(
                        floats[0], floats[1], floats[2], floats[3], floats[4], floats[5],
                        floats[6], floats[7], floats[8],
                    ),
                ))
            }
            "ConcatTransform" => {
                let floats = self.parse_floats()?;
                if floats.len() != 9 {
                    return Err(ParserError::Str(format!(
                        "{} parse_predirective(): ConcatTransform expects 9 floats but was given {}",
                        start_pos,
                        floats.len()
                    )));
                }
                Ok(PreDirective::ConcatTransform(
                    start_pos,
                    Matrix3f::new(
                        floats[0], floats[1], floats[2], floats[3], floats[4], floats[5],
                        floats[6], floats[7], floats[8],
                    ),
                ))
            }
            "ActiveTransform" => {
                let id = self.parse_identifier()?;
                Ok(PreDirective::ActiveTransform(start_pos, id))
            }
            "TransformTimes" => {
                let floats = self.parse_floats()?;
                if floats.len() != 2 {
                    return Err(ParserError::Str(format!(
                        "{} parse_predirective(): TransformTimes expects 2 floats but was given {}",
                        start_pos,
                        floats.len()
                    )));
                }
                Ok(PreDirective::TransformTimes(
                    start_pos, floats[0], floats[1],
                ))
            }
            "Camera" => {
                let ty = self.parse_string()?;
                self.parse_param_lists(&mut param_set)?;
                Ok(PreDirective::Camera(DirectiveStruct {
                    ty,
                    pos: start_pos,
                    param_set,
                }))
            }
            "Sampler" => {
                let ty = self.parse_string()?;
                self.parse_param_lists(&mut param_set)?;
                Ok(PreDirective::Sampler(DirectiveStruct {
                    ty,
                    pos: start_pos,
                    param_set,
                }))
            }
            "Film" => {
                let ty = self.parse_string()?;
                self.parse_param_lists(&mut param_set)?;
                Ok(PreDirective::Film(DirectiveStruct {
                    ty,
                    pos: start_pos,
                    param_set,
                }))
            }
            "Filter" => {
                let ty = self.parse_string()?;
                self.parse_param_lists(&mut param_set)?;
                Ok(PreDirective::Filter(DirectiveStruct {
                    ty,
                    pos: start_pos,
                    param_set,
                }))
            }
            "Integrator" => {
                let ty = self.parse_string()?;
                self.parse_param_lists(&mut param_set)?;
                Ok(PreDirective::Integrator(DirectiveStruct {
                    ty,
                    pos: start_pos,
                    param_set,
                }))
            }
            "Accelerator" => {
                let ty = self.parse_string()?;
                self.parse_param_lists(&mut param_set)?;
                Ok(PreDirective::Accelerator(DirectiveStruct {
                    ty,
                    pos: start_pos,
                    param_set,
                }))
            }
            "Include" => Ok(PreDirective::Include(
                start_pos,
                PathBuf::from(self.parse_string()?),
            )),
            id => Err(ParserError::Str(format!(
                "{} parse_predirective(): Unknown predirective: {}",
                start_pos, id
            ))),
        }
    }

    fn parse_predirectives(&mut self) -> ParserResult<Vec<PreDirective>> {
        let mut predirectives = vec![];
        loop {
            match self.peek()? {
                Token {
                    ty: TokenType::Identifier(ref s),
                    ..
                } => {
                    if s == "WorldBegin" {
                        return Ok(predirectives);
                    } else {
                        predirectives.push(self.parse_predirective()?);
                    }
                }
                Token { pos, ty } => {
                    return Err(ParserError::Str(format!(
                        "{} parse_predirectives(): expected predirective but got {:?}",
                        pos, ty
                    )));
                }
            }
        }
    }

    fn parse_spectrum(&mut self) -> ParserResult<Spectrum> {
        match self.peek() {
            Ok(Token {
                pos,
                ty: TokenType::LBracket,
            }) => {
                let floats = self.parse_list(Self::parse_float, "float")?;
                if floats.len() % 2 != 0 {
                    return Err(ParserError::Str(format!(
                        "{} parse_spectrum(): Expected an even number of floats but got {}",
                        pos,
                        floats.len()
                    )));
                }
                Ok(Spectrum::Spectrum(
                    floats.chunks(2).map(|chunk| (chunk[0], chunk[1])).collect(),
                ))
            }
            Ok(Token {
                ty: TokenType::Str(s),
                ..
            }) => {
                self.next()?;
                Ok(Spectrum::File(PathBuf::from(s)))
            }
            Ok(Token { pos, ty }) => Err(ParserError::Str(format!(
                "{pos} parse_spectrum(): expected '[' or string but got {ty:?}",
                pos = pos,
                ty = ty
            ))),
            Err(ParserError::Eof) => Err(ParserError::Str(
                "parse_spectrum(): EOF while processing spectrum".to_owned(),
            )),
            Err(err) => Err(err),
        }
    }

    pub fn parse(input: &str) -> ParserResult<(Vec<PreDirective>, Directive)> {
        let mut parser = Parser::new(input)?;
        Ok((parser.parse_predirectives()?, parser.parse_directive()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs::File;
    use std::io::prelude::*;
    use std::path::Path;

    #[test]
    fn test_tokenize_whitespace() {
        assert_eq!(Tokenizer::tokenize("   "), Ok(vec![]));
    }

    #[test]
    fn test_tokenize_comment() {
        assert_eq!(
            Tokenizer::tokenize("  # comment\n123"),
            Ok(vec![Token {
                pos: Pos::new(2, 1),
                ty: TokenType::Int(123),
            }])
        );
    }

    #[test]
    fn test_tokenize_positive_int() {
        assert_eq!(
            Tokenizer::tokenize("123"),
            Ok(vec![Token {
                pos: Pos::new(1, 1),
                ty: TokenType::Int(123),
            }])
        );
    }

    #[test]
    fn test_tokenize_negative_int() {
        assert_eq!(
            Tokenizer::tokenize("-123"),
            Ok(vec![Token {
                pos: Pos::new(1, 1),
                ty: TokenType::Int(-123),
            }])
        );
    }

    #[test]
    fn test_tokenize_negative_float() {
        assert_eq!(
            Tokenizer::tokenize("-1.23"),
            Ok(vec![Token {
                pos: Pos::new(1, 1),
                ty: TokenType::Float(-1.23),
            }])
        );
    }

    #[test]
    fn test_tokenize_positive_float() {
        assert_eq!(
            Tokenizer::tokenize("1.23"),
            Ok(vec![Token {
                pos: Pos::new(1, 1),
                ty: TokenType::Float(1.23),
            }])
        );
    }

    #[test]
    fn test_tokenize_exp1() {
        assert_eq!(
            Tokenizer::tokenize("1.23e12"),
            Ok(vec![Token {
                pos: Pos::new(1, 1),
                ty: TokenType::Float(1.23 * f64::powf(10.0, 12.0)),
            }])
        );
    }

    #[test]
    fn test_tokenize_exp2() {
        assert_eq!(
            Tokenizer::tokenize("1.23e-12"),
            Ok(vec![Token {
                pos: Pos::new(1, 1),
                ty: TokenType::Float(1.23 * f64::powf(10.0, -12.0)),
            }])
        );
    }

    #[test]
    fn test_tokenize_exp3() {
        assert_eq!(
            Tokenizer::tokenize("1e12"),
            Ok(vec![Token {
                pos: Pos::new(1, 1),
                ty: TokenType::Int(isize::pow(10, 12)),
            }])
        );
    }

    #[test]
    fn test_tokenize_exp4() {
        assert_eq!(
            Tokenizer::tokenize("1ed12"),
            Err(TokenizerError::Str(
                "(1, 3) tokenize_num(): no number".to_owned()
            ))
        );
    }

    #[test]
    fn test_tokenize1() {
        assert_eq!(
            Tokenizer::tokenize("Accelerator \"kdtree\" \"float emptybonus\" [0.1]"),
            Ok(vec![
                Token {
                    pos: Pos::new(1, 1),
                    ty: TokenType::Identifier("Accelerator".to_owned()),
                },
                Token {
                    pos: Pos::new(1, 13),
                    ty: TokenType::Str("kdtree".to_owned()),
                },
                Token {
                    pos: Pos::new(1, 22),
                    ty: TokenType::Str("float emptybonus".to_owned()),
                },
                Token {
                    pos: Pos::new(1, 41),
                    ty: TokenType::LBracket,
                },
                Token {
                    pos: Pos::new(1, 42),
                    ty: TokenType::Float(0.1),
                },
                Token {
                    pos: Pos::new(1, 45),
                    ty: TokenType::RBracket,
                },
            ])
        )
    }

    #[test]
    fn test_parse_one_int() {
        let mut parser = Parser::new("1").unwrap();
        assert_eq!(
            parser.parse_one_or_list(Parser::parse_int, "int"),
            Ok(vec![1])
        );
    }

    #[test]
    fn test_parse_ints() {
        let mut parser = Parser::new("[1 2 3]").unwrap();
        assert_eq!(
            parser.parse_one_or_list(Parser::parse_int, "int"),
            Ok(vec![1, 2, 3])
        );
    }

    #[test]
    fn test_parse_ints_err1() {
        let mut parser = Parser::new("[1 2 3.0]").unwrap();
        assert!(parser.parse_one_or_list(Parser::parse_int, "int").is_err());
    }

    #[test]
    fn test_parse_ints_err2() {
        let mut parser = Parser::new("[]").unwrap();
        assert!(parser.parse_one_or_list(Parser::parse_int, "int").is_err());
    }

    #[test]
    fn test_parse_one_bool() {
        let mut parser = Parser::new(r#"["true"]"#).unwrap();
        assert_eq!(
            parser.parse_one_or_list(Parser::parse_bool, "bool"),
            Ok(vec![true])
        );
    }

    #[test]
    fn test_parse_bools() {
        let mut parser = Parser::new(r#"["true" "false" "true"]"#).unwrap();
        assert_eq!(
            parser.parse_one_or_list(Parser::parse_bool, "bool"),
            Ok(vec![true, false, true])
        );
    }

    #[test]
    fn test_parse_bools_err() {
        let mut parser = Parser::new(r#"["true" "false" 3.0]"#).unwrap();
        assert!(
            parser
                .parse_one_or_list(Parser::parse_bool, "bool")
                .is_err()
        );
    }

    #[test]
    fn test_parse_one_float() {
        let mut parser = Parser::new("1.0").unwrap();
        assert_eq!(
            parser.parse_one_or_list(Parser::parse_float, "float"),
            Ok(vec![1.0])
        );
    }

    #[test]
    fn test_parse_floats() {
        let mut parser = Parser::new("[1 2.0 3]").unwrap();
        assert_eq!(
            parser.parse_one_or_list(Parser::parse_float, "float"),
            Ok(vec![1.0, 2.0, 3.0])
        );
    }

    #[test]
    fn test_parse_floats_err() {
        let mut parser = Parser::new("[1 test 2]").unwrap();
        assert!(
            parser
                .parse_one_or_list(Parser::parse_float, "float")
                .is_err()
        );
    }

    #[test]
    fn test_parse_one_point2() {
        let mut parser = Parser::new("1.0 2.0").unwrap();
        assert_eq!(
            parser.parse_one_or_list(Parser::parse_point2f, "point2f"),
            Ok(vec![Point2f::new(1.0, 2.0)])
        );
    }

    #[test]
    fn test_parse_point2s() {
        let mut parser = Parser::new("[1 2.0 3 4 5 6]").unwrap();
        assert_eq!(
            parser.parse_one_or_list(Parser::parse_point2f, "point2f"),
            Ok(vec![
                Point2f::new(1.0, 2.0),
                Point2f::new(3.0, 4.0),
                Point2f::new(5.0, 6.0),
            ])
        );
    }

    #[test]
    fn test_parse_point2s_err() {
        let mut parser = Parser::new("[1 2.0 3]").unwrap();
        let res = parser.parse_one_or_list(Parser::parse_point2f, "point2f");
        println!("{:?}", res);
        assert!(res.is_err());
    }

    #[test]
    fn test_parse_param_list1() {
        let mut parser = Parser::new(r#""float fov" [30]"#).unwrap();
        let mut param_set = ParamSet::default();
        assert_eq!(parser.parse_param_list(&mut param_set), Ok(()));
        assert_eq!(
            param_set.floats,
            vec![Param::new("fov", Pos::new(1, 1), vec![30.0])]
        );
    }

    #[test]
    fn test_parse_param_list2() {
        let mut parser = Parser::new(r#""point2 points" [1 2 3 4 5 6]"#).unwrap();
        let mut param_set = ParamSet::default();
        assert_eq!(parser.parse_param_list(&mut param_set), Ok(()));
        assert_eq!(
            param_set.point2fs,
            vec![Param::new(
                "points",
                Pos::new(1, 1),
                vec![
                    Point2f::new(1.0, 2.0),
                    Point2f::new(3.0, 4.0),
                    Point2f::new(5.0, 6.0),
                ],
            )]
        );
    }

    #[test]
    fn test_parse_param_list_rgb() {
        let mut parser = Parser::new(r#""rgb Kd" [.2 .5 .3]"#).unwrap();
        let mut param_set = ParamSet::default();
        assert_eq!(parser.parse_param_list(&mut param_set), Ok(()));
        assert_eq!(
            param_set.spectra,
            vec![Param::new(
                "Kd",
                Pos::new(1, 1),
                vec![Spectrum::Rgb(0.2, 0.5, 0.3)],
            )]
        );
    }

    #[test]
    fn test_parse_param_list_rgb_err() {
        let mut parser = Parser::new(r#""rgb Kd" [.2 .5]"#).unwrap();
        let mut param_set = ParamSet::default();
        assert!(parser.parse_param_list(&mut param_set).is_err());

        let mut parser = Parser::new(r#""rgb Kd" [.2 .5 .3 .4]"#).unwrap();
        let mut param_set = ParamSet::default();
        assert!(parser.parse_param_list(&mut param_set).is_err());
    }

    #[test]
    fn test_parse_param_list_spectrum() {
        let mut parser = Parser::new(r#""spectrum Kd" [300 .3 400 .6]"#).unwrap();
        let mut param_set = ParamSet::default();
        assert_eq!(parser.parse_param_list(&mut param_set), Ok(()));
        assert_eq!(
            param_set.spectra,
            vec![Param::new(
                "Kd",
                Pos::new(1, 1),
                vec![Spectrum::Spectrum(vec![(300.0, 0.3), (400.0, 0.6)])],
            )]
        );
    }

    #[test]
    fn test_parse_param_list_spectrum_file() {
        let mut parser = Parser::new(r#""spectrum Kd" "filename""#).unwrap();
        let mut param_set = ParamSet::default();
        assert_eq!(parser.parse_param_list(&mut param_set), Ok(()));
        assert_eq!(
            param_set.spectra,
            vec![Param::new(
                "Kd",
                Pos::new(1, 1),
                vec![Spectrum::File(PathBuf::from("filename"))],
            )]
        );
    }

    #[test]
    fn test_parse_directive() {
        let mut parser = Parser::new(r#"Shape "sphere" "float radius" 1"#).unwrap();
        let mut param_set = ParamSet::default();
        param_set
            .floats
            .push(Param::new("radius", Pos::new(1, 16), vec![1.0]));
        assert_eq!(
            parser.parse_directive(),
            Ok(Directive::Shape(DirectiveStruct {
                ty: "sphere".to_owned(),
                pos: Pos::new(1, 1),
                param_set
            }))
        );
    }

    #[test]
    fn test_parse_transform() {
        let mut parser = Parser::new(r#"Translate 1 0 0"#).unwrap();
        assert_eq!(
            parser.parse_predirective(),
            Ok(PreDirective::Translate(
                Pos::new(1, 1),
                Vector3f::new(1.0, 0.0, 0.0)
            ))
        );
    }

    #[test]
    fn test_parse_block() {
        let parser_test_dir = Path::new(
            &env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| r#"D:\projects\pbrtrs"#.to_owned()),
        ).join("parser_tests");
        let mut file = File::open(parser_test_dir.join("block.pbrt")).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        let mut parser = Parser::new(&contents).unwrap();
        assert_eq!(
            parser.parse_directive(),
            Ok(Directive::World(BlockStruct {
                pos: Pos::new(1, 1),
                children: vec![Directive::Attribute(BlockStruct {
                    pos: Pos::new(2, 5),
                    children: vec![
                        Directive::Material(DirectiveStruct {
                            ty: "glass".to_owned(),
                            pos: Pos::new(3, 9),
                            param_set: ParamSet::default(),
                        }),
                        Directive::Shape(DirectiveStruct {
                            ty: "sphere".to_owned(),
                            pos: Pos::new(4, 9),
                            param_set: ParamSet::default().add_floats(&[Param::new(
                                "radius",
                                Pos::new(4, 24),
                                vec![1.0],
                            )]),
                        }),
                    ],
                })],
            }))
        );
    }

    #[test]
    fn test_parse1() {
        let parser_test_dir = Path::new(
            &env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| r#"D:\projects\pbrtrs"#.to_owned()),
        ).join("parser_tests");
        let mut file = File::open(parser_test_dir.join("test1.pbrt")).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        assert_eq!(
            Parser::parse(&contents),
            Ok((
                vec![
                    PreDirective::Include(Pos::new(1, 1), PathBuf::from("file1.pbrt")),
                    PreDirective::LookAt(
                        Pos::new(3, 1),
                        Matrix3f::new(3.0, 4.0, 1.5, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0),
                    ),
                    PreDirective::Camera(DirectiveStruct {
                        pos: Pos::new(6, 1),
                        ty: "perspective".to_owned(),
                        param_set: ParamSet::default().add_floats(&[Param::new(
                            "fov",
                            Pos::new(6, 22),
                            vec![45.0],
                        )]),
                    }),
                    PreDirective::Sampler(DirectiveStruct {
                        pos: Pos::new(8, 1),
                        ty: "halton".to_owned(),
                        param_set: ParamSet::default().add_ints(&[Param::new(
                            "pixelsamples",
                            Pos::new(8, 18),
                            vec![128],
                        )]),
                    }),
                    PreDirective::Integrator(DirectiveStruct {
                        pos: Pos::new(9, 1),
                        ty: "path".to_owned(),
                        param_set: ParamSet::default(),
                    }),
                    PreDirective::Film(DirectiveStruct {
                        pos: Pos::new(10, 1),
                        ty: "image".to_owned(),
                        param_set: ParamSet::default()
                            .add_strings(&[Param::new(
                                "filename",
                                Pos::new(10, 14),
                                vec!["simple.png".to_owned()],
                            )])
                            .add_ints(&[
                                Param::new("xresolution", Pos::new(11, 6), vec![400]),
                                Param::new("yresolution", Pos::new(11, 34), vec![400]),
                            ]),
                    }),
                ],
                Directive::World(BlockStruct {
                    pos: Pos::new(13, 1),
                    children: vec![
                        Directive::LightSource(DirectiveStruct {
                            pos: Pos::new(14, 1),
                            ty: "infinite".to_owned(),
                            param_set: ParamSet::default().add_spectra(&[Param::new(
                                "L",
                                Pos::new(14, 24),
                                vec![Spectrum::Rgb(0.4, 0.45, 0.5)],
                            )]),
                        }),
                        Directive::Attribute(BlockStruct {
                            pos: Pos::new(16, 1),
                            children: vec![
                                Directive::Texture(TextureStruct {
                                    name: "checks".to_owned(),
                                    ty: "spectrum".to_owned(),
                                    class: "checkerboard".to_owned(),
                                    pos: Pos::new(17, 3),
                                    param_set: ParamSet::default()
                                        .add_floats(&[
                                            Param::new("uscale", Pos::new(18, 11), vec![8.0]),
                                            Param::new("vscale", Pos::new(18, 30), vec![8.0]),
                                        ])
                                        .add_spectra(&[
                                            Param::new(
                                                "tex1",
                                                Pos::new(19, 11),
                                                vec![Spectrum::Rgb(0.1, 0.1, 0.1)],
                                            ),
                                            Param::new(
                                                "tex2",
                                                Pos::new(19, 33),
                                                vec![Spectrum::Rgb(0.8, 0.8, 0.8)],
                                            ),
                                        ]),
                                }),
                                Directive::Material(DirectiveStruct {
                                    ty: "matte".to_owned(),
                                    pos: Pos::new(20, 3),
                                    param_set: ParamSet::default().add_textures(&[Param::new(
                                        "Kd",
                                        Pos::new(20, 20),
                                        vec!["checks".to_owned()],
                                    )]),
                                }),
                                Directive::Shape(DirectiveStruct {
                                    ty: "sphere".to_owned(),
                                    pos: Pos::new(21, 3),
                                    param_set: ParamSet::default().add_floats(&[Param::new(
                                        "radius",
                                        Pos::new(21, 18),
                                        vec![1.0],
                                    )]),
                                }),
                            ],
                        }),
                    ],
                })
            ))
        );
    }
}
