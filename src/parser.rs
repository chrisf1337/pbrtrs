use std::fmt;
use std::num::{ParseFloatError, ParseIntError};
use types::*;

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
struct Pos {
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

// pub struct Config {
//     before_transforms: Vec<Transform>,
//     after_transforms: Vec<Transform>,
//     transforms: Vec<Transform>,
// }

// pub enum Transform {
//     Identity,
//     Translate(f64, f64, f64),
//     Scale(f64, f64, f64),
// }

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
    fn add_float(mut self, p: Param<f64>) -> Self {
        {
            let this = &mut self;
            this.floats.push(p);
        }
        self
    }
}

#[derive(Debug, PartialEq)]
pub enum Directive {
    Material(DirectiveStruct),
    Shape(DirectiveStruct),
    Attribute(BlockStruct),
    Transform(BlockStruct),
}

#[derive(Debug, PartialEq)]
pub struct DirectiveStruct {
    ty: String,
    pos: Pos,
    param_set: ParamSet,
}

#[derive(Debug, PartialEq)]
pub struct BlockStruct {
    pos: Pos,
    children: Vec<Directive>,
}

#[derive(Debug, PartialEq, Eq)]
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
            Ok(c) if c == '-' || c == '+' || c.is_numeric() => self.tokenize_num(),
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
                            self.parse_one_or_list(Self::parse_ints)?,
                        ))
                    }
                    "float" => {
                        self.next()?;
                        param_set.floats.push(Param::new(
                            var,
                            pos,
                            self.parse_one_or_list(Self::parse_floats)?,
                        ))
                    }
                    "point2" => {
                        self.next()?;
                        param_set.point2fs.push(Param::new(
                            var,
                            pos,
                            self.parse_one_or_list(Self::parse_point2fs)?,
                        ))
                    }
                    "vector2" => {
                        self.next()?;
                        param_set.vector2fs.push(Param::new(
                            var,
                            pos,
                            self.parse_one_or_list(Self::parse_vector2fs)?,
                        ))
                    }
                    "bool" => {
                        self.next()?;
                        param_set.bools.push(Param::new(
                            var,
                            pos,
                            self.parse_one_or_list(Self::parse_bools)?,
                        ))
                    }
                    "string" => {
                        self.next()?;
                        param_set.strings.push(Param::new(
                            var,
                            pos,
                            self.parse_one_or_list(Self::parse_strings)?,
                        ))
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
            Ok(Token { pos, ty }) => {
                // no param list, so don't error out
                Ok(())
            }
            Err(ParserError::Eof) => Err(ParserError::Str(
                "parse_param_list(): EOF while processing param list".to_owned(),
            )),
            Err(err) => Err(err),
        }
    }

    fn parse_list<T>(
        &mut self,
        parser: fn(&mut Self) -> ParserResult<Vec<T>>,
    ) -> ParserResult<Vec<T>> {
        match self.peek() {
            Ok(Token {
                ty: TokenType::LBracket,
                ..
            }) => {
                let _ = self.next();
                let values = parser(self)?;
                match self.peek() {
                    Ok(Token {
                        ty: TokenType::RBracket,
                        ..
                    }) => Ok(values),
                    Ok(Token { pos, ty }) => Err(ParserError::Str(format!(
                        "{} parse_list(): expected ']' but got {:?}",
                        pos, ty
                    ))),
                    Err(ParserError::Eof) => Err(ParserError::Str(
                        "parse_list(): EOF while processing list".to_owned(),
                    )),
                    Err(err) => Err(err),
                }
            }
            Ok(Token { pos, ty }) => Err(ParserError::Str(format!(
                "{} parse_list(): expected '[' but got {:?}",
                pos, ty
            ))),
            Err(ParserError::Eof) => Err(ParserError::Str(
                "parse_list(): EOF while processing list".to_owned(),
            )),
            Err(err) => Err(err),
        }
    }

    fn parse_one_or_list<T>(
        &mut self,
        parser: fn(&mut Self) -> ParserResult<Vec<T>>,
    ) -> ParserResult<Vec<T>> {
        match self.peek() {
            Ok(Token {
                ty: TokenType::LBracket,
                ..
            }) => self.parse_list(parser),
            Ok(_) => parser(self),
            Err(ParserError::Eof) => Err(ParserError::Str(
                "parse_list(): EOF while processing list".to_owned(),
            )),
            Err(err) => Err(err),
        }
    }

    fn parse_ints(&mut self) -> ParserResult<Vec<isize>> {
        let mut values = vec![];
        loop {
            match self.peek() {
                Ok(Token {
                    ty: TokenType::Int(i),
                    ..
                }) => {
                    values.push(i);
                    let _ = self.next();
                }
                Ok(Token { pos, ty }) => {
                    if values.is_empty() {
                        return Err(ParserError::Str(format!(
                            "{} parse_ints(): expected int but got {:?}",
                            pos, ty
                        )));
                    } else {
                        return Ok(values);
                    }
                }
                Err(ParserError::Eof) => {
                    if values.is_empty() {
                        return Err(ParserError::Str(
                            "parse_ints(): EOF while processing ints".to_owned(),
                        ));
                    } else {
                        return Ok(values);
                    }
                }
                Err(err) => return Err(err),
            }
        }
    }

    fn parse_strings(&mut self) -> ParserResult<Vec<String>> {
        let mut values = vec![];
        loop {
            match self.peek() {
                Ok(Token {
                    ty: TokenType::Str(s),
                    ..
                }) => {
                    values.push(s);
                    let _ = self.next();
                }
                Ok(Token { pos, ty }) => {
                    if values.is_empty() {
                        return Err(ParserError::Str(format!(
                            "{} parse_strings(): expected string but got {:?}",
                            pos, ty
                        )));
                    } else {
                        return Ok(values);
                    }
                }
                Err(ParserError::Eof) => {
                    if values.is_empty() {
                        return Err(ParserError::Str(
                            "parse_strings(): EOF while processing strings".to_owned(),
                        ));
                    } else {
                        return Ok(values);
                    }
                }
                Err(err) => return Err(err),
            }
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

    fn parse_bools(&mut self) -> ParserResult<Vec<bool>> {
        let mut values = vec![];
        loop {
            match self.peek() {
                Ok(Token {
                    pos,
                    ty: TokenType::Str(s),
                }) => match s.as_ref() {
                    "true" => {
                        values.push(true);
                        let _ = self.next();
                    }
                    "false" => {
                        values.push(false);
                        let _ = self.next();
                    }
                    _ => {
                        return Err(ParserError::Str(format!(
                            "{} parse_bools(): expected bool but got {:?}",
                            pos, s
                        )))
                    }
                },
                Ok(Token { pos, ty }) => {
                    if values.is_empty() {
                        return Err(ParserError::Str(format!(
                            "{} parse_bools(): expected bool but got {:?}",
                            pos, ty
                        )));
                    } else {
                        return Ok(values);
                    }
                }
                Err(ParserError::Eof) => {
                    if values.is_empty() {
                        return Err(ParserError::Str(
                            "parse_bools(): EOF while processing bools".to_owned(),
                        ));
                    } else {
                        return Ok(values);
                    }
                }
                Err(err) => return Err(err),
            }
        }
    }

    fn parse_floats(&mut self) -> ParserResult<Vec<f64>> {
        let mut values = vec![];
        loop {
            match self.peek() {
                Ok(Token {
                    ty: TokenType::Float(f),
                    ..
                }) => {
                    values.push(f);
                    let _ = self.next();
                }
                Ok(Token {
                    ty: TokenType::Int(i),
                    ..
                }) => {
                    values.push(i as f64);
                    let _ = self.next();
                }
                Ok(Token { pos, ty }) => {
                    if values.is_empty() {
                        return Err(ParserError::Str(format!(
                            "{} parse_floats(): expected float but got {:?}",
                            pos, ty
                        )));
                    } else {
                        return Ok(values);
                    }
                }
                Err(ParserError::Eof) => {
                    if values.is_empty() {
                        return Err(ParserError::Str(
                            "parse_floats(): EOF while processing floats".to_owned(),
                        ));
                    } else {
                        return Ok(values);
                    }
                }
                Err(err) => return Err(err),
            }
        }
    }

    fn parse_point2fs(&mut self) -> ParserResult<Vec<Point2f>> {
        let start_pos = self.pos()?;
        let floats = self.parse_floats()?;
        // parse_floats() errors out if there are no floats, so we're guaranteed at least one float here
        if floats.len() % 2 != 0 {
            return Err(ParserError::Str(format!(
                "{} parse_point2fs(): expected an even number of floats but got {}",
                start_pos,
                floats.len()
            )));
        }
        Ok(floats
            .chunks(2)
            .map(|fs| Point2f::new(fs[0], fs[1]))
            .collect())
    }

    fn parse_vector2fs(&mut self) -> ParserResult<Vec<Vector2f>> {
        let start_pos = self.pos()?;
        let floats = self.parse_floats()?;
        // parse_floats() errors out if there are no floats, so we're guaranteed at least one float here
        if floats.len() % 2 != 0 {
            return Err(ParserError::Str(format!(
                "{} parse_vector2fs(): expected an even number of floats but got {}",
                start_pos,
                floats.len()
            )));
        }
        Ok(floats
            .chunks(2)
            .map(|fs| Vector2f::new(fs[0], fs[1]))
            .collect())
    }

    fn parse_directive(&mut self) -> ParserResult<Directive> {
        let start_pos = self.pos()?;
        let id = self.parse_identifier()?;
        let mut param_set = ParamSet::default();
        match id.as_ref() {
            "Material" => {
                let ty = self.parse_string()?;
                self.parse_param_list(&mut param_set)?;
                Ok(Directive::Material(DirectiveStruct {
                    ty,
                    pos: start_pos,
                    param_set,
                }))
            }
            "Shape" => {
                let ty = self.parse_string()?;
                self.parse_param_list(&mut param_set)?;
                Ok(Directive::Shape(DirectiveStruct {
                    ty,
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
                    if s == "AttributeEnd" || s == "TransformEnd" {
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

    fn parse_block(&mut self) -> ParserResult<Directive> {
        let start_pos = self.pos()?;
        match self.parse_identifier()?.as_ref() {
            "AttributeBegin" => Ok(Directive::Attribute(BlockStruct {
                pos: start_pos,
                children: self.parse_directives()?,
            })),
            "TransformBegin" => Ok(Directive::Transform(BlockStruct {
                pos: start_pos,
                children: self.parse_directives()?,
            })),
            id => Err(ParserError::Str(format!(
                "{} parse_block(): expected identifier or block start but got {:?}",
                start_pos, id
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs::File;
    use std::io::prelude::*;
    use std::path::{Path, PathBuf};

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
        assert_eq!(parser.parse_one_or_list(Parser::parse_ints), Ok(vec![1]));
    }

    #[test]
    fn test_parse_ints() {
        let mut parser = Parser::new("[1 2 3]").unwrap();
        assert_eq!(
            parser.parse_one_or_list(Parser::parse_ints),
            Ok(vec![1, 2, 3])
        );
    }

    #[test]
    fn test_parse_ints_err1() {
        let mut parser = Parser::new("[1 2 3.0]").unwrap();
        assert!(parser.parse_one_or_list(Parser::parse_ints).is_err());
    }

    #[test]
    fn test_parse_ints_err2() {
        let mut parser = Parser::new("[]").unwrap();
        assert!(parser.parse_one_or_list(Parser::parse_ints).is_err());
    }

    #[test]
    fn test_parse_one_bool() {
        let mut parser = Parser::new(r#"["true"]"#).unwrap();
        assert_eq!(
            parser.parse_one_or_list(Parser::parse_bools),
            Ok(vec![true])
        );
    }

    #[test]
    fn test_parse_bools() {
        let mut parser = Parser::new(r#"["true" "false" "true"]"#).unwrap();
        assert_eq!(
            parser.parse_one_or_list(Parser::parse_bools),
            Ok(vec![true, false, true])
        );
    }

    #[test]
    fn test_parse_bools_err() {
        let mut parser = Parser::new(r#"["true" "false" 3.0]"#).unwrap();
        assert!(parser.parse_one_or_list(Parser::parse_bools).is_err());
    }

    #[test]
    fn test_parse_one_float() {
        let mut parser = Parser::new("1.0").unwrap();
        assert_eq!(
            parser.parse_one_or_list(Parser::parse_floats),
            Ok(vec![1.0])
        );
    }

    #[test]
    fn test_parse_floats() {
        let mut parser = Parser::new("[1 2.0 3]").unwrap();
        assert_eq!(
            parser.parse_one_or_list(Parser::parse_floats),
            Ok(vec![1.0, 2.0, 3.0])
        );
    }

    #[test]
    fn test_parse_floats_err() {
        let mut parser = Parser::new("[1 test 2]").unwrap();
        assert!(parser.parse_one_or_list(Parser::parse_floats).is_err());
    }

    #[test]
    fn test_parse_one_point2() {
        let mut parser = Parser::new("1.0 2.0").unwrap();
        assert_eq!(
            parser.parse_one_or_list(Parser::parse_point2fs),
            Ok(vec![Point2f::new(1.0, 2.0)])
        );
    }

    #[test]
    fn test_parse_point2s() {
        let mut parser = Parser::new("[1 2.0 3 4 5 6]").unwrap();
        assert_eq!(
            parser.parse_one_or_list(Parser::parse_point2fs),
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
        assert!(parser.parse_one_or_list(Parser::parse_point2fs).is_err());
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
    fn test_parse_block() {
        let parser_test_dir = Path::new(
            &env::var("CARGO_MANIFEST_DIR").unwrap_or(r#"D:\projects\pbrtrs"#.to_owned()),
        ).join("parser_tests");
        let mut file = File::open(parser_test_dir.join("test1.pbrt")).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        let mut parser = Parser::new(&contents).unwrap();
        assert_eq!(
            parser.parse_block(),
            Ok(Directive::Attribute(BlockStruct {
                pos: Pos::new(1, 1),
                children: vec![
                    Directive::Material(DirectiveStruct {
                        ty: "glass".to_owned(),
                        pos: Pos::new(2, 5),
                        param_set: ParamSet::default(),
                    }),
                    Directive::Shape(DirectiveStruct {
                        ty: "sphere".to_owned(),
                        pos: Pos::new(3, 5),
                        param_set: ParamSet::default().add_float(Param::new(
                            "radius",
                            Pos::new(3, 20),
                            vec![1.0],
                        )),
                    }),
                ],
            }))
        );
    }
}
