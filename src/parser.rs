use std::fmt;
use std::num::{ParseFloatError, ParseIntError};

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

pub struct Config {
    before_transforms: Vec<Transform>,
    after_transforms: Vec<Transform>,
    transforms: Vec<Transform>,
}

pub enum Transform {
    Identity,
    Translate(f64, f64, f64),
    Scale(f64, f64, f64),
}

pub struct Param<T> {
    name: String,
    values: Vec<T>,
}

pub enum ParamType {
    Bool,
    Int,
    Float,
    Point2,
    Point3,
    Vector3,
    Normal3,
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
            pos: Pos::new(0, 0),
            prev_pos: Pos::new(0, 0),
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
            Pos::new(self.pos.r + 1, 0)
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

    // fn parse_param(&mut self) -> ParserResult<Param> {
    //     match self.peek() {
    //         Ok(Token { pos, ty: TokenType::Str(s) }) => {
    //             let split = s.split_whitespace();
    //             if split.len() != 2 {
    //                 return Err(ParserError::Str(format!("({}, {}) parse_param(): expecting two words in string, got {}", pos.0, pos.1, s)));
    //             }
    //             match split[0].as_ref() {
    //                 "int" =>
    //             }
    //         }
    //         Ok(tok) => Err(ParserError::Str(format!("({}, {}) parse_param(): expecting string, got {:?}", tok.pos.0, tok.pos.1, tok))),
    //         Err(ParserError::Eof) =>
    //                 Err(ParserError::Str("tokenize_str(): EOF while processing string".to_owned())),
    //         Err(err) => Err(err),
    //     }
    // }

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

    // fn parse_point2s(&mut self)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_whitespace() {
        assert_eq!(Tokenizer::tokenize("   "), Ok(vec![]));
    }

    #[test]
    fn test_tokenize_comment() {
        assert_eq!(
            Tokenizer::tokenize("  # comment\n123"),
            Ok(vec![Token {
                pos: Pos::new(1, 0),
                ty: TokenType::Int(123),
            }])
        );
    }

    #[test]
    fn test_tokenize_positive_int() {
        assert_eq!(
            Tokenizer::tokenize("123"),
            Ok(vec![Token {
                pos: Pos::new(0, 0),
                ty: TokenType::Int(123),
            }])
        );
    }

    #[test]
    fn test_tokenize_negative_int() {
        assert_eq!(
            Tokenizer::tokenize("-123"),
            Ok(vec![Token {
                pos: Pos::new(0, 0),
                ty: TokenType::Int(-123),
            }])
        );
    }

    #[test]
    fn test_tokenize_negative_float() {
        assert_eq!(
            Tokenizer::tokenize("-1.23"),
            Ok(vec![Token {
                pos: Pos::new(0, 0),
                ty: TokenType::Float(-1.23),
            }])
        );
    }

    #[test]
    fn test_tokenize_positive_float() {
        assert_eq!(
            Tokenizer::tokenize("1.23"),
            Ok(vec![Token {
                pos: Pos::new(0, 0),
                ty: TokenType::Float(1.23),
            }])
        );
    }

    #[test]
    fn test_tokenize_exp1() {
        assert_eq!(
            Tokenizer::tokenize("1.23e12"),
            Ok(vec![Token {
                pos: Pos::new(0, 0),
                ty: TokenType::Float(1.23 * f64::powf(10.0, 12.0)),
            }])
        );
    }

    #[test]
    fn test_tokenize_exp2() {
        assert_eq!(
            Tokenizer::tokenize("1.23e-12"),
            Ok(vec![Token {
                pos: Pos::new(0, 0),
                ty: TokenType::Float(1.23 * f64::powf(10.0, -12.0)),
            }])
        );
    }

    #[test]
    fn test_tokenize_exp3() {
        assert_eq!(
            Tokenizer::tokenize("1e12"),
            Ok(vec![Token {
                pos: Pos::new(0, 0),
                ty: TokenType::Int(isize::pow(10, 12)),
            }])
        );
    }

    #[test]
    fn test_tokenize_exp4() {
        assert_eq!(
            Tokenizer::tokenize("1ed12"),
            Err(TokenizerError::Str(
                "(0, 2) tokenize_num(): no number".to_owned()
            ))
        );
    }

    #[test]
    fn test_tokenize1() {
        assert_eq!(
            Tokenizer::tokenize("Accelerator \"kdtree\" \"float emptybonus\" [0.1]"),
            Ok(vec![
                Token {
                    pos: Pos::new(0, 0),
                    ty: TokenType::Identifier("Accelerator".to_owned()),
                },
                Token {
                    pos: Pos::new(0, 12),
                    ty: TokenType::Str("kdtree".to_owned()),
                },
                Token {
                    pos: Pos::new(0, 21),
                    ty: TokenType::Str("float emptybonus".to_owned()),
                },
                Token {
                    pos: Pos::new(0, 40),
                    ty: TokenType::LBracket,
                },
                Token {
                    pos: Pos::new(0, 41),
                    ty: TokenType::Float(0.1),
                },
                Token {
                    pos: Pos::new(0, 44),
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
    fn test_parse_ints_err() {
        let mut parser = Parser::new("[1 2 3.0]").unwrap();
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
}
